#pragma once
//
// Minimal process-isolated test framework.
//
// Every test runs in its own forked child process, so a segfault, an abort or
// an ASan report in one test is reported as a failure for THAT test instead of
// taking the whole suite down with it. That is the point of this harness: the
// code under test is known to crash, and we still want a full result table.
//

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <unistd.h>
#include <sys/wait.h>
#include <csignal>
#include <cerrno>

namespace testing
{
    struct TestCase
    {
        const char* name;
        void (*fn)();
    };

    inline std::vector<TestCase>& registry()
    {
        static std::vector<TestCase> tests;
        return tests;
    }

    struct Registrar
    {
        Registrar(const char* name, void (*fn)())
        {
            registry().push_back(TestCase{name, fn});
        }
    };

    // Thrown by the assertion macros. Caught inside the child process.
    struct Failure
    {
        std::string message;
    };

    inline bool nearlyEqual(double a, double b, double tolerance)
    {
        if (std::isnan(a) || std::isnan(b)) return false;
        return std::fabs(a - b) <= tolerance;
    }

    // ---- child-side reporting -------------------------------------------------

    inline int runInChild(const TestCase& test)
    {
        fflush(nullptr); // don't duplicate buffered parent output into the child
        pid_t pid = fork();
        if (pid < 0)
        {
            std::printf("  fork() failed: %s\n", std::strerror(errno));
            return 1;
        }
        if (pid == 0)
        {
            alarm(10); // a hung test is a failed test
            try
            {
                test.fn();
            }
            catch (const Failure& f)
            {
                std::printf("%s", f.message.c_str());
                fflush(nullptr);
                _exit(1);
            }
            catch (const std::exception& e)
            {
                std::printf("      unexpected std::exception: %s\n", e.what());
                fflush(nullptr);
                _exit(1);
            }
            catch (...)
            {
                std::printf("      unexpected unknown exception\n");
                fflush(nullptr);
                _exit(1);
            }
            fflush(nullptr);
            _exit(0);
        }

        int status = 0;
        waitpid(pid, &status, 0);
        if (WIFSIGNALED(status))
        {
            int sig = WTERMSIG(status);
            std::printf("      process died: signal %d (%s)\n", sig, strsignal(sig));
            return 2; // crashed
        }
        if (WIFEXITED(status) && WEXITSTATUS(status) != 0)
            return 1; // failed (or ASan exited non-zero)
        return 0;
    }

    inline int runAll(const char* suiteName, const char* filter)
    {
        std::printf("\n=== %s ===\n", suiteName);
        int passed = 0, failed = 0, crashed = 0, skipped = 0;

        for (const TestCase& test : registry())
        {
            if (filter && *filter && std::strstr(test.name, filter) == nullptr)
            {
                ++skipped;
                continue;
            }
            std::printf("[ RUN      ] %s\n", test.name);
            fflush(nullptr);
            int rc = runInChild(test);
            if (rc == 0)      { std::printf("[       OK ] %s\n", test.name);      ++passed; }
            else if (rc == 2) { std::printf("[    CRASH ] %s\n", test.name);      ++crashed; }
            else              { std::printf("[     FAIL ] %s\n", test.name);      ++failed; }
            fflush(nullptr);
        }

        std::printf("\n--- summary ---\n");
        std::printf("passed:  %d\n", passed);
        std::printf("failed:  %d\n", failed);
        std::printf("crashed: %d\n", crashed);
        if (skipped) std::printf("skipped: %d (filter \"%s\")\n", skipped, filter);
        std::printf("\n");
        return (failed + crashed) == 0 ? 0 : 1;
    }
}

// ---- registration ------------------------------------------------------------

#define TEST(name)                                                             \
    static void name();                                                        \
    static ::testing::Registrar registrar_##name(#name, name);                 \
    static void name()

// ---- assertions --------------------------------------------------------------
// Each formats a detailed message and throws, unwinding to the child's handler.

#define TEST_FAIL_(...)                                                        \
    do {                                                                       \
        char buf_[1024];                                                       \
        std::snprintf(buf_, sizeof(buf_), __VA_ARGS__);                        \
        char msg_[1400];                                                       \
        std::snprintf(msg_, sizeof(msg_), "      %s:%d\n      %s\n",           \
                      __FILE__, __LINE__, buf_);                               \
        throw ::testing::Failure{msg_};                                        \
    } while (0)

#define ASSERT_TRUE(cond)                                                      \
    do {                                                                       \
        if (!(cond)) TEST_FAIL_("expected true: %s", #cond);                    \
    } while (0)

#define ASSERT_FALSE(cond)                                                     \
    do {                                                                       \
        if ((cond)) TEST_FAIL_("expected false: %s", #cond);                    \
    } while (0)

#define ASSERT_EQ_INT(actual, expected)                                        \
    do {                                                                       \
        long long a_ = (long long)(actual);                                     \
        long long e_ = (long long)(expected);                                   \
        if (a_ != e_)                                                          \
            TEST_FAIL_("%s == %s\n        actual:   %lld\n        expected: %lld", \
                       #actual, #expected, a_, e_);                            \
    } while (0)

#define ASSERT_NEAR(actual, expected, tol)                                     \
    do {                                                                       \
        double a_ = (double)(actual);                                          \
        double e_ = (double)(expected);                                        \
        if (!::testing::nearlyEqual(a_, e_, (tol)))                            \
            TEST_FAIL_("%s ~= %s\n        actual:   %.12g\n        expected: %.12g  (tol %g)", \
                       #actual, #expected, a_, e_, (double)(tol));             \
    } while (0)

// Same as ASSERT_NEAR but names the index being compared, for vector checks.
#define ASSERT_NEAR_AT(idx, actual, expected, tol)                             \
    do {                                                                       \
        double a_ = (double)(actual);                                          \
        double e_ = (double)(expected);                                        \
        if (!::testing::nearlyEqual(a_, e_, (tol)))                            \
            TEST_FAIL_("at index %d: %s ~= %s\n        actual:   %.12g\n        expected: %.12g  (tol %g)", \
                       (int)(idx), #actual, #expected, a_, e_, (double)(tol)); \
    } while (0)

#define ASSERT_FINITE(value)                                                   \
    do {                                                                       \
        double v_ = (double)(value);                                           \
        if (!std::isfinite(v_))                                                \
            TEST_FAIL_("expected finite: %s (was %.12g)", #value, v_);         \
    } while (0)

// Documents WHY a failure matters, right where the reader sees it.
#define FAIL_WITH_NOTE(note, ...)                                              \
    do {                                                                       \
        char detail_[900];                                                     \
        std::snprintf(detail_, sizeof(detail_), __VA_ARGS__);                  \
        TEST_FAIL_("%s\n      note: %s", detail_, note);                       \
    } while (0)
