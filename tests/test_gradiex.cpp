// GradieX test suite.
// Includes main.cpp directly (renaming its main) so the tests exercise the real
// implementation with no duplicated source.
//
// Build & run:
//   g++ -std=c++17 -O2 -Wall -Wextra -pedantic tests/test_gradiex.cpp -o test_gradiex && ./test_gradiex
#define main gradiex_example_main
#include "../main.cpp"
#undef main
#include <functional>
#include <sstream>

struct Hush {
    streambuf* saved;
    ostringstream sink;
    Hush() : saved(cout.rdbuf()) { cout.rdbuf(sink.rdbuf()); }
    ~Hush() { cout.rdbuf(saved); }
};

static int g_pass = 0, g_fail = 0;

// invalid_argument specifically: an API-contract rejection.
static bool throws(const std::function<void()>& fn) {
    try { fn(); return false; } catch (const invalid_argument&) { return true; } catch (...) { return false; }
}
// any exception: used for I/O failures, which surface as runtime_error.
static bool throws_any(const std::function<void()>& fn) {
    try { fn(); return false; } catch (...) { return true; }
}

void check(bool ok, const string& name, const string& detail = "") {
    if (ok) { ++g_pass; cout << "  [PASS] " << name; }
    else    { ++g_fail; cout << "  [FAIL] " << name; }
    if (!detail.empty()) cout << "  (" << detail << ")";
    cout << endl;
}

// ---- helpers -------------------------------------------------------------
vector<double*> collect_params(NeuralNetwork& net) {
    vector<double*> p;
    for (Layer& layer : net.hidden_layers)
        for (Neuron& n : layer.neurons) {
            for (double& w : n.weights) p.push_back(&w);
            p.push_back(&n.bias);
        }
    for (Neuron& n : net.output_layer.neurons) {
        for (double& w : n.weights) p.push_back(&w);
        p.push_back(&n.bias);
    }
    return p;
}

vector<double> collect_grads(NeuralNetwork& net) {
    vector<double> g;
    for (Layer& layer : net.hidden_layers)
        for (Neuron& n : layer.neurons) {
            for (double v : n.weight_gradients) g.push_back(v);
            g.push_back(n.bias_gradient);
        }
    for (Neuron& n : net.output_layer.neurons) {
        for (double v : n.weight_gradients) g.push_back(v);
        g.push_back(n.bias_gradient);
    }
    return g;
}

// Three linearly-separable-ish Gaussian blobs, one-hot encoded.
void make_three_class(vector<vector<double>>& X, vector<vector<double>>& Y, int per_class) {
    X.clear(); Y.clear();
    mt19937 rng(24601);
    normal_distribution<double> jitter(0.0, 0.32);
    double centres[3][2] = {{0.0, 1.6}, {-1.5, -0.9}, {1.5, -0.9}};
    for (int label = 0; label < 3; ++label)
        for (int i = 0; i < per_class; ++i) {
            X.push_back({centres[label][0] + jitter(rng), centres[label][1] + jitter(rng)});
            vector<double> one_hot(3, 0.0);
            one_hot[label] = 1.0;
            Y.push_back(one_hot);
        }
}

void set_deterministic_weights(NeuralNetwork& net, unsigned seed) {
    mt19937 rng(seed);
    uniform_real_distribution<double> dist(-0.9, 0.9);
    for (double* p : collect_params(net)) *p = dist(rng);
}

// ---- TEST 1: numerical gradient check -----------------------------------
// Verifies backprop against central-difference numerical gradients.
// backward_pass with learning_rate=1.0 gives grad = (w_before - w_after).
double gradient_check(const string& label, int in_size, int n_hidden, int width,
                      int out_size, const string& hid_act, const string& out_act,
                      const string& loss, const vector<double>& x,
                      const vector<double>& t, unsigned seed) {
    NeuralNetwork net(in_size, n_hidden, width, out_size, hid_act, out_act, loss);
    set_deterministic_weights(net, seed);
    vector<double*> params = collect_params(net);

    net.forward_pass(x);
    vector<double> before;
    for (double* p : params) before.push_back(*p);

    net.backward_pass(t, 1.0);

    vector<double> analytic;
    for (size_t i = 0; i < params.size(); ++i) analytic.push_back(before[i] - *params[i]);
    for (size_t i = 0; i < params.size(); ++i) *params[i] = before[i];

    const double h = 1e-5;
    double worst = 0.0, worst_abs = 0.0;
    int tiny = 0;
    for (size_t i = 0; i < params.size(); ++i) {
        *params[i] = before[i] + h;
        double lp = net.calculate_loss(net.forward_pass(x), t);
        *params[i] = before[i] - h;
        double lm = net.calculate_loss(net.forward_pass(x), t);
        *params[i] = before[i];

        double numeric = (lp - lm) / (2.0 * h);
        double abs_err = fabs(analytic[i] - numeric);
        worst_abs = max(worst_abs, abs_err);
        // Gradients this small are below the resolution of a central difference
        // on an O(1) loss; judge them on absolute error only.
        if (max(fabs(analytic[i]), fabs(numeric)) < 1e-6) { ++tiny; continue; }
        worst = max(worst, abs_err / (fabs(analytic[i]) + fabs(numeric)));
    }
    cout << "  " << left << setw(46) << label << " params=" << setw(4) << params.size()
         << " max_rel_err=" << scientific << setprecision(3) << worst
         << " max_abs_err=" << worst_abs;
    if (tiny) cout << "  (" << tiny << " near-zero grads by abs err)";
    cout << endl;
    if (worst_abs > 1e-9) worst = max(worst, 1.0);  // fail loudly if abs err is large too
    return worst;
}

int main() {
    cout << fixed;
    cout << "=========================================================" << endl;
    cout << " TEST 1: Backprop numerical gradient check (tol 1e-5)" << endl;
    cout << "=========================================================" << endl;
    double tol = 1e-5, worst_overall = 0.0;
    struct Case { string label, hid, out, loss; int in, nh, w, o; vector<double> x, t; };
    vector<Case> cases = {
        {"tanh + sigmoid + BCE (XOR config)", "tanh","sigmoid","binary_cross_entropy",       2,2,4,1,{1.0,0.0},{1.0}},
        {"sigmoid + sigmoid + BCE",           "sigmoid","sigmoid","binary_cross_entropy",    3,2,5,1,{0.5,-1.0,2.0},{0.0}},
        {"ReLU + linear + MSE",               "ReLU","linear","mean_squared_error",          3,2,4,2,{0.7,1.3,-0.4},{1.0,-2.0}},
        {"leaky_ReLU + linear + MSE",         "leaky_ReLU","linear","mean_squared_error",    4,3,5,3,{-1.0,0.2,0.9,-0.3},{0.5,1.5,-1.0}},
        {"tanh + linear + MSE (deep, 4 hidden)","tanh","linear","mean_squared_error",        3,4,6,2,{0.3,-0.8,1.1},{2.0,0.0}},
        {"tanh + sigmoid + BCE (multi-output)","tanh","sigmoid","binary_cross_entropy",      3,2,5,3,{1.0,-0.5,0.25},{1.0,0.0,1.0}},
        {"no hidden layers (0 hidden)",       "tanh","sigmoid","binary_cross_entropy",       3,0,4,1,{0.4,0.9,-1.2},{1.0}},
        {"tanh + softmax + CCE (3 classes)",  "tanh","softmax","categorical_cross_entropy",  3,2,5,3,{0.5,-1.0,0.7},{0.0,1.0,0.0}},
        {"ReLU + softmax + CCE (4 cls, deep)","ReLU","softmax","categorical_cross_entropy",  4,3,6,4,{0.8,-0.3,1.2,0.1},{0.0,0.0,1.0,0.0}},
        {"tanh + softmax + CCE (2 classes)",  "tanh","softmax","categorical_cross_entropy",  2,1,4,2,{-0.6,1.4},{1.0,0.0}},
    };
    for (size_t i = 0; i < cases.size(); ++i) {
        const Case& c = cases[i];
        double e = gradient_check(c.label, c.in, c.nh, c.w, c.o, c.hid, c.out, c.loss,
                                 c.x, c.t, 1000u + static_cast<unsigned>(i));
        worst_overall = max(worst_overall, e);
    }
    cout << endl;
    check(worst_overall < tol, "All gradient checks within tolerance",
          "worst rel err over all configs = " + to_string(worst_overall));

    cout << "\n=========================================================" << endl;
    cout << " TEST 2: Activation functions & derivatives" << endl;
    cout << "=========================================================" << endl;
    check(fabs(Neuron::activate(0.0, "sigmoid") - 0.5) < 1e-12, "sigmoid(0) == 0.5");
    check(Neuron::activate(-800.0, "sigmoid") >= 0.0 && Neuron::activate(800.0, "sigmoid") <= 1.0
          && !std::isnan(Neuron::activate(-800.0, "sigmoid")),
          "sigmoid numerically stable at +/-800 (no NaN/overflow)");
    check(fabs(Neuron::activate(0.5, "tanh") - tanh(0.5)) < 1e-12, "tanh matches std::tanh");
    check(Neuron::activate(-3.0, "ReLU") == 0.0 && Neuron::activate(3.0, "ReLU") == 3.0, "ReLU clamps negatives");
    check(fabs(Neuron::activate(-2.0, "leaky_ReLU") + 0.02) < 1e-12, "leaky_ReLU(-2) == -0.02");
    check(Neuron::activate(7.5, "linear") == 7.5, "linear is identity");
    // derivatives vs numerical
    double dh = 1e-7; bool dok = true; string dbad;
    for (const string& act : {string("sigmoid"), string("tanh"), string("ReLU"), string("linear"), string("leaky_ReLU")})
        for (double v : {-2.3, -0.7, 0.4, 1.8}) {
            double num = (Neuron::activate(v+dh, act) - Neuron::activate(v-dh, act)) / (2*dh);
            double ana = Neuron::activation_derivative(v, Neuron::activate(v, act), act);
            if (fabs(num - ana) > 1e-5) { dok = false; dbad += act + "@" + to_string(v) + " "; }
        }
    check(dok, "All activation derivatives match numerical diff", dbad.empty() ? "20 point checks" : dbad);

    cout << "\n=========================================================" << endl;
    cout << " TEST 3: Loss functions" << endl;
    cout << "=========================================================" << endl;
    NeuralNetwork mse_net(2,1,2,1,"tanh","linear","mean_squared_error");
    check(fabs(mse_net.calculate_loss({3.0,1.0},{1.0,1.0}) - 2.0) < 1e-12, "MSE({3,1},{1,1}) == 2.0");
    check(mse_net.calculate_loss({5.0},{5.0}) == 0.0, "MSE of perfect prediction == 0");
    NeuralNetwork bce_net(2,1,2,1,"tanh","sigmoid","binary_cross_entropy");
    check(fabs(bce_net.calculate_loss({0.5},{1.0}) - (-log(0.5))) < 1e-9, "BCE(0.5, t=1) == ln2");
    check(bce_net.calculate_loss({1.0},{1.0}) < 1e-6, "BCE(1.0, t=1) ~ 0 (clamped, no inf)");
    check(std::isfinite(bce_net.calculate_loss({0.0},{1.0})), "BCE(0.0, t=1) finite (epsilon clamp works)");

    cout << "\n=========================================================" << endl;
    cout << " TEST 4: Forward pass structure" << endl;
    cout << "=========================================================" << endl;
    NeuralNetwork shape_net(4,3,7,5,"ReLU","sigmoid","binary_cross_entropy");
    vector<double> out = shape_net.forward_pass({1.0,2.0,3.0,4.0});
    check(out.size() == 5, "Output width == 5", "got " + to_string(out.size()));
    check(shape_net.hidden_layers.size() == 3, "Hidden layer count == 3");
    bool widths_ok = true;
    for (Layer& l : shape_net.hidden_layers) if (l.neurons.size() != 7) widths_ok = false;
    check(widths_ok, "Every hidden layer has 7 neurons");
    check(shape_net.hidden_layers[0].neurons[0].weights.size() == 4, "Layer 1 fan-in == input size (4)");
    check(shape_net.hidden_layers[1].neurons[0].weights.size() == 7, "Layer 2 fan-in == prev width (7)");
    check(shape_net.output_layer.neurons[0].weights.size() == 7, "Output fan-in == last hidden width (7)");
    bool range_ok = true;
    for (double v : out) if (v < 0.0 || v > 1.0) range_ok = false;
    check(range_ok, "Sigmoid outputs all within [0,1]");
    vector<double> a = shape_net.forward_pass({1.0,2.0,3.0,4.0});
    vector<double> b = shape_net.forward_pass({1.0,2.0,3.0,4.0});
    check(a == b, "Forward pass is deterministic (same input -> same output)");

    cout << "\n=========================================================" << endl;
    cout << " TEST 5: Error handling" << endl;
    cout << "=========================================================" << endl;

    check(throws([]{ NeuralNetwork(0,1,2,1,"tanh","sigmoid","mean_squared_error"); }), "Rejects input_size = 0");
    check(throws([]{ NeuralNetwork(2,1,0,1,"tanh","sigmoid","mean_squared_error"); }), "Rejects layer_width = 0");
    check(throws([]{ NeuralNetwork(2,1,2,0,"tanh","sigmoid","mean_squared_error"); }), "Rejects output_width = 0");
    check(throws([]{ NeuralNetwork(2,-1,2,1,"tanh","sigmoid","mean_squared_error"); }), "Rejects negative hidden count");
    check(throws([]{ Neuron::activate(1.0, "swish"); }), "Rejects unknown activation");
    check(throws([]{ NeuralNetwork n(2,1,2,1,"tanh","sigmoid","huber"); n.calculate_loss({0.5},{1.0}); }), "Rejects unknown loss");
    check(throws([]{ NeuralNetwork n(2,1,2,1,"tanh","sigmoid","mean_squared_error"); n.forward_pass({1.0,2.0,3.0}); }), "Rejects wrong input dimension");
    check(throws([]{ NeuralNetwork n(2,1,2,1,"tanh","sigmoid","mean_squared_error"); n.forward_pass({1.0,2.0}); n.calculate_loss(n.output_layer.output_vector,{1.0,2.0}); }), "Rejects target/prediction size mismatch");
    check(throws([]{ NeuralNetwork n(2,1,2,1,"tanh","sigmoid","mean_squared_error"); n.train({{1,2}},{{1}},0,0.1); }), "Rejects epochs = 0");
    check(throws([]{ NeuralNetwork n(2,1,2,1,"tanh","sigmoid","mean_squared_error"); n.train({{1,2}},{{1}},10,-0.1); }), "Rejects negative learning rate");
    check(throws([]{ NeuralNetwork n(2,1,2,1,"tanh","sigmoid","mean_squared_error"); n.train({},{},10,0.1); }), "Rejects empty dataset");
    check(throws([]{ NeuralNetwork n(2,1,2,1,"tanh","sigmoid","mean_squared_error"); n.train({{1,2}},{{1},{0}},10,0.1); }), "Rejects input/target count mismatch");

    cout << "\n=========================================================" << endl;
    cout << " TEST 6: Learning tasks (loss must actually decrease)" << endl;
    cout << "=========================================================" << endl;
    // 6a: XOR convergence rate across 40 random inits
    int solved = 0; const int trials = 40;
    for (int i = 0; i < trials; ++i) {
        NeuralNetwork n(2,2,4,1,"tanh","sigmoid","binary_cross_entropy");
        { Hush h; n.train({{0,0},{0,1},{1,0},{1,1}}, {{0},{1},{1},{0}}, 5000, 0.1, 100000); }
        bool ok = n.predict({0,0})[0] < 0.1 && n.predict({0,1})[0] > 0.9
               && n.predict({1,0})[0] > 0.9 && n.predict({1,1})[0] < 0.1;
        if (ok) ++solved;
    }
    cout << "  XOR solved in " << solved << "/" << trials << " random inits" << endl;
    check(solved >= trials * 8 / 10, "XOR converges in >=80% of random inits");

    // 6b: linear regression y = 2*x1 - 3*x2 + 0.5, MSE + linear output
    {
        vector<vector<double>> X, Y;
        mt19937 rng(7); uniform_real_distribution<double> d(-1.0, 1.0);
        for (int i = 0; i < 200; ++i) {
            double x1 = d(rng), x2 = d(rng);
            X.push_back({x1, x2}); Y.push_back({2.0*x1 - 3.0*x2 + 0.5});
        }
        NeuralNetwork n(2,2,8,1,"tanh","linear","mean_squared_error");
        double before = 0.0;
        for (size_t i = 0; i < X.size(); ++i) before += n.calculate_loss(n.forward_pass(X[i]), Y[i]);
        before /= X.size();
        { Hush h; n.train(X, Y, 300, 0.02, 100000); }
        double after = 0.0;
        for (size_t i = 0; i < X.size(); ++i) after += n.calculate_loss(n.forward_pass(X[i]), Y[i]);
        after /= X.size();
        cout << "  Regression MSE: " << fixed << setprecision(6) << before << " -> " << after << endl;
        check(after < before * 0.05 && after < 0.01, "Regression (MSE+linear) loss drops >95%");
    }

    // 6c: ReLU hidden + multi-output classification
    {
        vector<vector<double>> X = {{0,0},{0,1},{1,0},{1,1}};
        vector<vector<double>> Y = {{0,0},{1,0},{1,0},{0,1}};  // XOR, AND
        NeuralNetwork n(2,2,16,2,"ReLU","sigmoid","binary_cross_entropy");
        double before = 0.0;
        for (size_t i = 0; i < X.size(); ++i) before += n.calculate_loss(n.forward_pass(X[i]), Y[i]);
        before /= X.size();
        { Hush h; n.train(X, Y, 4000, 0.05, 100000); }
        double after = 0.0;
        for (size_t i = 0; i < X.size(); ++i) after += n.calculate_loss(n.forward_pass(X[i]), Y[i]);
        after /= X.size();
        cout << "  Multi-output (XOR,AND) BCE: " << before << " -> " << after << endl;
        check(after < before * 0.2, "Multi-output ReLU net (width 16) loss drops >80%");
    }

    // 6d: loss is monotonically non-increasing overall (sanity on update direction)
    {
        NeuralNetwork n(3,2,6,1,"tanh","sigmoid","binary_cross_entropy");
        vector<vector<double>> X = {{0.1,0.2,0.3},{0.9,0.8,0.7},{0.4,0.1,0.9},{0.2,0.7,0.4}};
        vector<vector<double>> Y = {{0},{1},{1},{0}};
        double prev = 1e18; int increases = 0;
        for (int e = 0; e < 400; ++e) {
            double l = 0.0;
            for (size_t i = 0; i < X.size(); ++i) {
                l += n.calculate_loss(n.forward_pass(X[i]), Y[i]);
                n.backward_pass(Y[i], 0.05);
            }
            l /= X.size();
            if (l > prev + 1e-9) ++increases;
            prev = l;
        }
        cout << "  Epochs where loss increased: " << increases << "/400, final loss " << prev << endl;
        check(increases <= 20 && prev < 0.05, "Training descends steadily (<=5% of epochs increase)");
    }

    cout << "\n=========================================================" << endl;
    cout << " TEST 7: Numerical robustness" << endl;
    cout << "=========================================================" << endl;
    {
        NeuralNetwork n(3,3,6,2,"tanh","sigmoid","binary_cross_entropy");
        vector<double> big = {1e5, -1e5, 1e5};
        vector<double> o = n.forward_pass(big);
        bool fin = true; for (double v : o) if (!std::isfinite(v)) fin = false;
        check(fin, "Large inputs (1e5) produce finite outputs");
        n.backward_pass({1.0,0.0}, 0.01);
        bool pfin = true; for (double* p : collect_params(n)) if (!std::isfinite(*p)) pfin = false;
        check(pfin, "Params remain finite after backward on large inputs");
    }
    {
        NeuralNetwork n(2,2,4,1,"ReLU","sigmoid","binary_cross_entropy");
        { Hush h; n.train({{0,0},{0,1},{1,0},{1,1}}, {{0},{1},{1},{0}}, 2000, 0.1, 100000); }
        bool pfin = true; for (double* p : collect_params(n)) if (!std::isfinite(*p)) pfin = false;
        check(pfin, "No NaN/Inf in weights after 2000 ReLU epochs");
    }
    {
        // He init sanity: variance ~ 2/fan_in
        NeuralNetwork n(100, 1, 500, 1, "ReLU", "sigmoid", "binary_cross_entropy");
        double sum = 0.0, sumsq = 0.0; int cnt = 0;
        for (Neuron& nu : n.hidden_layers[0].neurons)
            for (double w : nu.weights) { sum += w; sumsq += w*w; ++cnt; }
        double var = sumsq/cnt - (sum/cnt)*(sum/cnt);
        double expected = 2.0/100.0;
        cout << "  He init variance: " << setprecision(5) << var << " (expected ~" << expected << ")" << endl;
        check(fabs(var - expected) / expected < 0.10, "He initialization variance == 2/fan_in (+/-10%)");
        bool biases_zero = true;
        for (Neuron& nu : n.hidden_layers[0].neurons) if (nu.bias != 0.0) biases_zero = false;
        check(biases_zero, "Biases initialized to zero");
    }
    {
        // distinct random init per network instance
        NeuralNetwork n1(4,1,4,1,"tanh","sigmoid","mean_squared_error");
        NeuralNetwork n2(4,1,4,1,"tanh","sigmoid","mean_squared_error");
        check(n1.hidden_layers[0].neurons[0].weights != n2.hidden_layers[0].neurons[0].weights,
              "Separate instances get different random weights");
    }

    cout << "\n=========================================================" << endl;
    cout << " TEST 8: Dying-ReLU characterization (known ReLU property)" << endl;
    cout << "=========================================================" << endl;
    {
        auto xor_rate = [](const string& act, int width, int trials) {
            int solved = 0;
            for (int i = 0; i < trials; ++i) {
                NeuralNetwork n(2,2,width,1,act,"sigmoid","binary_cross_entropy", 1000 + i);
                { Hush h; n.train({{0,0},{0,1},{1,0},{1,1}},{{0},{1},{1},{0}},4000,0.05,1000000); }
                if (n.predict({0,0})[0]<0.1 && n.predict({0,1})[0]>0.9
                 && n.predict({1,0})[0]>0.9 && n.predict({1,1})[0]<0.1) ++solved;
            }
            return solved;
        };
        const int N = 60;
        int r4  = xor_rate("ReLU", 4, N);
        int r16 = xor_rate("ReLU", 16, N);
        int l4  = xor_rate("leaky_ReLU", 4, N);
        cout << "  ReLU w4: " << r4 << "/" << N << "   ReLU w16: " << r16 << "/" << N
             << "   leaky_ReLU w4: " << l4 << "/" << N << endl;
        // Bounds derived from a 200-init measurement: ReLU/16 100%, leaky/4 90.5%, ReLU/4 67.5%.
        check(r16 >= 57, "Wide ReLU (16) solves XOR reliably (measured 100%)", to_string(r16) + "/" + to_string(N));
        check(l4 >= 45, "leaky_ReLU at width 4 mostly avoids dead units (measured 90.5%)", to_string(l4) + "/" + to_string(N));
        check(r4 >= 30, "Narrow ReLU still converges most of the time (measured 67.5%)", to_string(r4) + "/" + to_string(N));
        check(r16 > r4, "Failure rate falls as ReLU width grows => dying ReLU, not a gradient bug",
              "w4=" + to_string(r4) + " vs w16=" + to_string(r16));
    }

    cout << "\n=========================================================" << endl;
    cout << " TEST 9: Seed control & reproducibility" << endl;
    cout << "=========================================================" << endl;
    {
        NeuralNetwork a(3,2,5,2,"tanh","sigmoid","binary_cross_entropy", 12345);
        NeuralNetwork b(3,2,5,2,"tanh","sigmoid","binary_cross_entropy", 12345);
        NeuralNetwork c(3,2,5,2,"tanh","sigmoid","binary_cross_entropy", 99999);
        check(a.seed == 12345u, "Network reports the seed it was given");
        vector<double*> pa = collect_params(a), pb = collect_params(b), pc = collect_params(c);
        bool same = true, diff = false;
        for (size_t i = 0; i < pa.size(); ++i) {
            if (*pa[i] != *pb[i]) same = false;
            if (*pa[i] != *pc[i]) diff = true;
        }
        check(same, "Same seed => bit-identical initial weights");
        check(diff, "Different seed => different initial weights");
        check(a.predict({0.3,0.4,0.5}) == b.predict({0.3,0.4,0.5}), "Same seed => identical predictions");

        // Full training determinism, with shuffling AND dropout both active.
        vector<vector<double>> X = {{0,0},{0,1},{1,0},{1,1},{0.5,0.2},{0.2,0.8},{0.9,0.1},{0.4,0.6}};
        vector<vector<double>> Y = {{0},{1},{1},{0},{1},{1},{1},{0}};
        TrainOptions o; o.epochs = 60; o.learning_rate = 0.05; o.batch_size = 3;
        o.shuffle = true; o.dropout = 0.25; o.verbose = false;
        NeuralNetwork t1(2,2,8,1,"tanh","sigmoid","binary_cross_entropy", 777);
        NeuralNetwork t2(2,2,8,1,"tanh","sigmoid","binary_cross_entropy", 777);
        t1.train(X, Y, o); t2.train(X, Y, o);
        vector<double*> q1 = collect_params(t1), q2 = collect_params(t2);
        bool trained_same = true;
        for (size_t i = 0; i < q1.size(); ++i) if (*q1[i] != *q2[i]) trained_same = false;
        check(trained_same, "Same seed => identical weights after shuffled+dropout training");

        NeuralNetwork r1(3,1,4,1,"tanh","sigmoid","mean_squared_error");
        NeuralNetwork r2(3,1,4,1,"tanh","sigmoid","mean_squared_error");
        check(r1.seed != r2.seed, "Default (unseeded) construction still varies per instance");
    }

    cout << "\n=========================================================" << endl;
    cout << " TEST 10: Mini-batch gradient correctness" << endl;
    cout << "=========================================================" << endl;
    {
        vector<vector<double>> X = {{0.4,-0.7,1.1},{-0.2,0.9,0.3},{1.0,0.1,-0.5},{0.3,0.6,0.8}};
        vector<vector<double>> Y = {{1.0,0.0},{0.0,1.0},{1.0,1.0},{0.0,0.0}};
        const double lr = 0.1;

        NeuralNetwork net(3,2,5,2,"tanh","sigmoid","binary_cross_entropy", 4242);
        vector<double*> P = collect_params(net);
        vector<double> w0; for (double* p : P) w0.push_back(*p);

        // Per-sample gradients, averaged by hand.
        vector<double> mean_grad(P.size(), 0.0);
        for (size_t s_i = 0; s_i < X.size(); ++s_i) {
            for (size_t i = 0; i < P.size(); ++i) *P[i] = w0[i];
            net.zero_gradients();
            net.forward_pass(X[s_i]);
            net.accumulate_gradients(Y[s_i]);
            vector<double> g = collect_grads(net);
            for (size_t i = 0; i < g.size(); ++i) mean_grad[i] += g[i] / static_cast<double>(X.size());
        }

        // One batch update over all four samples.
        for (size_t i = 0; i < P.size(); ++i) *P[i] = w0[i];
        net.zero_gradients();
        for (size_t s_i = 0; s_i < X.size(); ++s_i) {
            net.forward_pass(X[s_i]);
            net.accumulate_gradients(Y[s_i]);
        }
        net.apply_gradients(lr, 1.0 / static_cast<double>(X.size()), 0.0);

        double worst = 0.0;
        for (size_t i = 0; i < P.size(); ++i)
            worst = max(worst, fabs(*P[i] - (w0[i] - lr * mean_grad[i])));
        cout << "  max deviation from hand-averaged per-sample gradients: "
             << scientific << setprecision(2) << worst << endl;
        check(worst < 1e-15, "Batch update == mean of per-sample gradients");

        // Legacy signature must remain bit-identical to a manual per-sample loop.
        NeuralNetwork la(3,2,5,2,"tanh","sigmoid","binary_cross_entropy", 555);
        NeuralNetwork lb(3,2,5,2,"tanh","sigmoid","binary_cross_entropy", 555);
        { Hush h; la.train(X, Y, 25, 0.1, 100000); }
        for (int e = 0; e < 25; ++e)
            for (size_t s_i = 0; s_i < X.size(); ++s_i) {
                lb.forward_pass(X[s_i]);
                lb.backward_pass(Y[s_i], 0.1);
            }
        vector<double*> qa = collect_params(la), qb = collect_params(lb);
        bool legacy_same = true;
        for (size_t i = 0; i < qa.size(); ++i) if (*qa[i] != *qb[i]) legacy_same = false;
        check(legacy_same, "Legacy train() still == manual fixed-order per-sample loop");

        // Ragged final batch: 4 samples with batch_size 3 => batches of 3 and 1.
        NeuralNetwork ragged(3,1,4,2,"tanh","sigmoid","binary_cross_entropy", 31);
        TrainOptions ro; ro.epochs = 5; ro.learning_rate = 0.05; ro.batch_size = 3;
        ro.shuffle = false; ro.verbose = false;
        bool ok_ragged = true;
        try { ragged.train(X, Y, ro); } catch (...) { ok_ragged = false; }
        bool finite_after = true;
        for (double* p : collect_params(ragged)) if (!std::isfinite(*p)) finite_after = false;
        check(ok_ragged && finite_after, "Ragged final batch handled (4 samples, batch 3)");

        NeuralNetwork big(3,1,4,2,"tanh","sigmoid","binary_cross_entropy", 32);
        TrainOptions bo; bo.epochs = 5; bo.learning_rate = 0.05; bo.batch_size = 999;
        bo.shuffle = false; bo.verbose = false;
        bool ok_big = true;
        try { big.train(X, Y, bo); } catch (...) { ok_big = false; }
        check(ok_big, "batch_size larger than dataset => single full batch");
        check(throws([&]{ NeuralNetwork n(3,1,4,2,"tanh","sigmoid","binary_cross_entropy",1);
                          TrainOptions z; z.batch_size = 0; z.verbose = false; n.train(X,Y,z); }),
              "Rejects batch_size = 0");
    }

    cout << "\n=========================================================" << endl;
    cout << " TEST 11: Shuffling" << endl;
    cout << "=========================================================" << endl;
    {
        vector<vector<double>> X, Y;
        mt19937 rg(3); uniform_real_distribution<double> d(-1,1);
        for (int i = 0; i < 40; ++i) { double a = d(rg), b = d(rg);
            X.push_back({a,b}); Y.push_back({(a*b > 0) ? 1.0 : 0.0}); }

        TrainOptions on;  on.epochs = 30; on.learning_rate = 0.05; on.batch_size = 8;
        on.shuffle = true;  on.verbose = false;
        TrainOptions off; off.epochs = 30; off.learning_rate = 0.05; off.batch_size = 8;
        off.shuffle = false; off.verbose = false;

        NeuralNetwork s1(2,2,6,1,"tanh","sigmoid","binary_cross_entropy", 606);
        NeuralNetwork s2(2,2,6,1,"tanh","sigmoid","binary_cross_entropy", 606);
        NeuralNetwork s3(2,2,6,1,"tanh","sigmoid","binary_cross_entropy", 606);
        s1.train(X, Y, on); s2.train(X, Y, on); s3.train(X, Y, off);

        vector<double*> a = collect_params(s1), b = collect_params(s2), c = collect_params(s3);
        bool shuffled_same = true, order_matters = false;
        for (size_t i = 0; i < a.size(); ++i) {
            if (*a[i] != *b[i]) shuffled_same = false;
            if (*a[i] != *c[i]) order_matters = true;
        }
        check(shuffled_same, "Shuffled training is reproducible under a fixed seed");
        check(order_matters, "shuffle=on and shuffle=off diverge (order really changes)");
    }

    cout << "\n=========================================================" << endl;
    cout << " TEST 12: Softmax + categorical cross-entropy" << endl;
    cout << "=========================================================" << endl;
    {
        NeuralNetwork sm(4,2,6,5,"tanh","softmax","categorical_cross_entropy", 8080);
        vector<double> p = sm.forward_pass({0.5,-1.0,0.25,0.75});
        double sum = 0.0; bool in_range = true;
        for (double v : p) { sum += v; if (v < 0.0 || v > 1.0) in_range = false; }
        check(fabs(sum - 1.0) < 1e-12, "Softmax outputs sum to 1", "sum = " + to_string(sum));
        check(in_range, "Softmax outputs all within [0,1]");

        vector<double> big = sm.forward_pass({1e5,-1e5,1e5,-1e5});
        double bsum = 0.0; bool bfin = true;
        for (double v : big) { bsum += v; if (!std::isfinite(v)) bfin = false; }
        check(bfin && fabs(bsum - 1.0) < 1e-12, "Softmax stable at +/-1e5 logits (max-shift works)",
              "sum = " + to_string(bsum));

        check(throws([]{ NeuralNetwork(3,2,4,3,"softmax","softmax","categorical_cross_entropy"); }),
              "Rejects softmax as a hidden activation");
        check(throws([]{ NeuralNetwork(3,2,4,3,"tanh","softmax","binary_cross_entropy"); }),
              "Rejects softmax paired with a non-CCE loss");
        check(throws([]{ NeuralNetwork(3,2,4,3,"tanh","sigmoid","categorical_cross_entropy"); }),
              "Rejects CCE without a softmax output");
        check(throws([]{ Neuron::activate(1.0, "softmax"); }),
              "Neuron::activate refuses softmax (layer-wide activation)");
        check(throws([]{ Neuron::activation_derivative(1.0, 0.5, "softmax"); }),
              "Softmax derivative refuses the diagonal path (fused with loss)");

        // Learns a 3-class problem.
        vector<vector<double>> X, Y;
        make_three_class(X, Y, 60);
        NeuralNetwork clf(2,2,16,3,"tanh","softmax","categorical_cross_entropy", 1234);
        double before = clf.evaluate(X, Y);
        TrainOptions o; o.epochs = 250; o.learning_rate = 0.05; o.batch_size = 12;
        o.shuffle = true; o.verbose = false;
        clf.train(X, Y, o);
        double after = clf.evaluate(X, Y), acc = clf.accuracy(X, Y);
        cout << "  3-class CCE: " << fixed << setprecision(4) << before << " -> " << after
             << ", accuracy " << setprecision(1) << 100.0*acc << "%" << endl;
        check(after < before * 0.2, "Softmax net drives CCE down >80%");
        check(acc > 0.95, "Softmax net reaches >95% accuracy", to_string(100.0*acc) + "%");
    }

    cout << "\n=========================================================" << endl;
    cout << " TEST 13: Save / load" << endl;
    cout << "=========================================================" << endl;
    {
        const string path  = "/tmp/gradiex_test_model.gx";
        const string path2 = "/tmp/gradiex_test_model2.gx";

        NeuralNetwork orig(4,3,7,3,"leaky_ReLU","softmax","categorical_cross_entropy", 5150);
        vector<vector<double>> X, Y;
        make_three_class(X, Y, 40);
        // Reshape the 2-feature set to 4 features so it matches this net.
        for (vector<double>& row : X) { row.push_back(row[0]*row[1]); row.push_back(row[0]-row[1]); }
        TrainOptions o; o.epochs = 40; o.learning_rate = 0.05; o.batch_size = 8; o.verbose = false;
        orig.train(X, Y, o);
        orig.save(path);

        NeuralNetwork back = NeuralNetwork::load(path);
        check(back.hidden_layers.size() == orig.hidden_layers.size(), "Hidden layer count survives save/load");
        check(back.output_size() == orig.output_size(), "Output width survives save/load");
        check(back.loss == orig.loss, "Loss function survives save/load");
        check(back.output_layer.activation == Activation::Softmax, "Output activation survives save/load");
        check(back.hidden_layers[0].activation == Activation::LeakyReLU, "Hidden activation survives save/load");
        check(back.input_size() == orig.input_size(), "Input width survives save/load");
        check(back.seed == orig.seed, "Seed survives save/load");

        bool bit_exact = true;
        for (const vector<double>& row : X) {
            vector<double> a = orig.predict(row), b = back.predict(row);
            if (a != b) bit_exact = false;
        }
        check(bit_exact, "Reloaded model predicts bit-for-bit identically on all inputs");
        check(fabs(orig.accuracy(X,Y) - back.accuracy(X,Y)) < 1e-15, "Reloaded accuracy identical");

        back.save(path2);
        ifstream f1(path, ios::binary), f2(path2, ios::binary);
        string c1((istreambuf_iterator<char>(f1)), istreambuf_iterator<char>());
        string c2((istreambuf_iterator<char>(f2)), istreambuf_iterator<char>());
        check(c1 == c2 && !c1.empty(), "save -> load -> save is byte-identical",
              to_string(c1.size()) + " bytes");

        // A trained model reloaded mid-flight can keep training.
        double before_more = back.evaluate(X, Y);
        TrainOptions more; more.epochs = 40; more.learning_rate = 0.05; more.batch_size = 8; more.verbose = false;
        back.train(X, Y, more);
        check(back.evaluate(X, Y) < before_more, "Reloaded model can resume training");

        { ofstream bad("/tmp/gradiex_bad.gx"); bad << "not a model at all\n"; }
        check(throws_any([]{ NeuralNetwork::load("/tmp/gradiex_bad.gx"); }), "Rejects a non-model file");
        { ofstream v("/tmp/gradiex_ver.gx"); v << "GRADIEX-MODEL 99\n"; }
        check(throws_any([]{ NeuralNetwork::load("/tmp/gradiex_ver.gx"); }), "Rejects an unsupported version");
        { ofstream t("/tmp/gradiex_trunc.gx");
          t << "GRADIEX-MODEL 1\nloss mean_squared_error\nseed 1\nlayers 1\nlayer 0 2 2 tanh\n0.1 0.2 0.3\n"; }
        check(throws_any([]{ NeuralNetwork::load("/tmp/gradiex_trunc.gx"); }), "Rejects a truncated model file");
        check(throws_any([]{ NeuralNetwork::load("/tmp/definitely_absent_dir_xyz/m.gx"); }), "Rejects a missing file");
        check(throws_any([]{ NeuralNetwork n(2,1,2,1,"tanh","sigmoid","mean_squared_error",1);
                             n.save("/tmp/definitely_absent_dir_xyz/m.gx"); }), "save() reports an unwritable path");
    }

    cout << "\n=========================================================" << endl;
    cout << " TEST 14: L2 regularization" << endl;
    cout << "=========================================================" << endl;
    {
        // With zero gradients, an L2 step is pure exponential shrinkage.
        NeuralNetwork net(3,1,4,1,"tanh","sigmoid","binary_cross_entropy", 90);
        vector<double*> P = collect_params(net);
        vector<double> before; for (double* p : P) before.push_back(*p);
        const double lr = 0.1, l2 = 0.05;
        net.zero_gradients();
        net.apply_gradients(lr, 1.0, l2);

        bool shrink_exact = true, bias_untouched = true;
        size_t idx = 0;
        for (Layer& L : net.hidden_layers)
            for (Neuron& n : L.neurons) {
                for (size_t w = 0; w < n.weights.size(); ++w, ++idx)
                    if (fabs(n.weights[w] - before[idx] * (1.0 - lr*l2)) > 1e-15) shrink_exact = false;
                if (n.bias != before[idx]) bias_untouched = false;
                ++idx;
            }
        check(shrink_exact, "Zero-gradient L2 step scales weights by exactly (1 - lr*l2)");
        check(bias_untouched, "L2 does not decay biases");

        // Same data and seed: more L2 => smaller weight norm.
        vector<vector<double>> X, Y;
        mt19937 rg(11); uniform_real_distribution<double> d(-1,1);
        for (int i = 0; i < 60; ++i) { double a = d(rg), b = d(rg);
            X.push_back({a,b}); Y.push_back({(a+b > 0) ? 1.0 : 0.0}); }
        auto norm_after = [&](double l2v) {
            NeuralNetwork n(2,2,10,1,"tanh","sigmoid","binary_cross_entropy", 4711);
            TrainOptions o; o.epochs = 120; o.learning_rate = 0.05; o.batch_size = 10;
            o.shuffle = true; o.l2 = l2v; o.verbose = false;
            n.train(X, Y, o);
            double total = 0.0;
            for (Layer& L : n.hidden_layers) for (Neuron& nu : L.neurons)
                for (double w : nu.weights) total += w*w;
            for (Neuron& nu : n.output_layer.neurons) for (double w : nu.weights) total += w*w;
            return sqrt(total);
        };
        double n0 = norm_after(0.0), n1 = norm_after(0.01);
        cout << "  weight L2 norm: l2=0 -> " << fixed << setprecision(3) << n0
             << ",  l2=0.01 -> " << n1 << endl;
        check(n1 < n0, "L2 > 0 yields a smaller trained weight norm");
        check(throws([&]{ NeuralNetwork n(2,1,3,1,"tanh","sigmoid","binary_cross_entropy",1);
                          TrainOptions o; o.l2 = -1.0; o.verbose = false; n.train(X,Y,o); }),
              "Rejects negative L2");
    }

    cout << "\n=========================================================" << endl;
    cout << " TEST 15: Dropout" << endl;
    cout << "=========================================================" << endl;
    {
        NeuralNetwork net(3,2,40,1,"tanh","sigmoid","binary_cross_entropy", 2468);
        vector<double> x = {0.6,-0.3,0.9};

        net.forward_pass(x, false, 0.5);
        check(net.hidden_layers[0].dropout_mask.empty(), "Inference mode applies no dropout mask");
        vector<double> clean = net.forward_pass(x);
        check(net.forward_pass(x) == clean, "Inference is deterministic with dropout configured");

        net.forward_pass(x, true, 0.5);
        const Layer& L0 = net.hidden_layers[0];
        int zeros = 0; bool scale_ok = true;
        for (size_t i = 0; i < L0.neurons.size(); ++i) {
            if (L0.dropout_mask[i] == 0.0) { ++zeros; if (L0.output_vector[i] != 0.0) scale_ok = false; }
            else if (fabs(L0.dropout_mask[i] - 2.0) > 1e-12) scale_ok = false;
        }
        cout << "  dropout p=0.5 on a 40-unit layer: " << zeros << " units dropped" << endl;
        check(zeros > 5 && zeros < 35, "Training mode drops roughly half the units");
        check(scale_ok, "Survivors scaled by 1/(1-p) = 2; dropped units output exactly 0");

        // Inverted dropout must preserve the expected activation.
        vector<double> sum(L0.neurons.size(), 0.0);
        const int trials = 4000;
        for (int t = 0; t < trials; ++t) {
            net.forward_pass(x, true, 0.5);
            for (size_t i = 0; i < sum.size(); ++i) sum[i] += net.hidden_layers[0].output_vector[i];
        }
        net.forward_pass(x);
        double worst_rel = 0.0;
        for (size_t i = 0; i < sum.size(); ++i) {
            double expected = net.hidden_layers[0].output_vector[i];
            if (fabs(expected) < 1e-3) continue;
            worst_rel = max(worst_rel, fabs(sum[i]/trials - expected) / fabs(expected));
        }
        cout << "  worst relative gap between E[dropout] and the clean activation: "
             << fixed << setprecision(4) << worst_rel << endl;
        check(worst_rel < 0.08, "Inverted dropout preserves the expected activation");

        // A dropped unit must receive no gradient.
        NeuralNetwork g(3,1,30,1,"tanh","sigmoid","binary_cross_entropy", 1357);
        g.zero_gradients();
        g.forward_pass(x, true, 0.5);
        g.accumulate_gradients({1.0});
        bool dropped_have_no_grad = true, kept_have_grad = false;
        Layer& GL = g.hidden_layers[0];
        for (size_t i = 0; i < GL.neurons.size(); ++i) {
            double total = fabs(GL.neurons[i].bias_gradient);
            for (double gr : GL.neurons[i].weight_gradients) total += fabs(gr);
            if (GL.dropout_mask[i] == 0.0) { if (total != 0.0) dropped_have_no_grad = false; }
            else if (total > 0.0) kept_have_grad = true;
        }
        check(dropped_have_no_grad, "Dropped units accumulate exactly zero gradient");
        check(kept_have_grad, "Surviving units still accumulate gradient");
        check(throws([&]{ NeuralNetwork n(2,1,3,1,"tanh","sigmoid","binary_cross_entropy",1);
                          vector<vector<double>> X={{0,0}}, Y={{0}};
                          TrainOptions o; o.dropout = 1.0; o.verbose = false; n.train(X,Y,o); }),
              "Rejects dropout rate >= 1");
    }

    cout << "\n=========================================================" << endl;
    cout << " TEST 16: Train / validation split" << endl;
    cout << "=========================================================" << endl;
    {
        vector<vector<double>> X, Y;
        make_three_class(X, Y, 40);   // 120 points
        NeuralNetwork net(2,2,12,3,"tanh","softmax","categorical_cross_entropy", 31337);
        TrainOptions o; o.epochs = 80; o.learning_rate = 0.05; o.batch_size = 10;
        o.shuffle = true; o.validation_split = 0.25; o.verbose = false;
        TrainHistory h = net.train(X, Y, o);

        check(h.training_samples == 90 && h.validation_samples == 30,
              "0.25 split of 120 => 90 train / 30 validation",
              to_string(h.training_samples) + "/" + to_string(h.validation_samples));
        check(h.training_samples + h.validation_samples == static_cast<int>(X.size()),
              "Split covers the dataset exactly, no overlap or loss");
        check(static_cast<int>(h.train_loss.size()) == o.epochs, "Train-loss history has one entry per epoch");
        check(static_cast<int>(h.validation_loss.size()) == o.epochs, "Validation-loss history has one entry per epoch");
        check(h.best_epoch >= 1 && h.best_epoch <= o.epochs, "best_epoch is inside the run");
        check(h.best_validation_loss <= h.validation_loss.front(), "Validation loss improved over the run");
        check(h.train_loss.back() < h.train_loss.front(), "Train loss decreased over the run");

        TrainHistory h0 = net.train(X, Y, [&]{ TrainOptions z = o; z.validation_split = 0.0; z.epochs = 5; return z; }());
        check(h0.validation_samples == 0 && h0.validation_loss.empty(),
              "No split => no validation set and no validation history");
        check(throws([&]{ NeuralNetwork n(2,1,4,3,"tanh","softmax","categorical_cross_entropy",1);
                          TrainOptions z; z.validation_split = 1.0; z.verbose = false; n.train(X,Y,z); }),
              "Rejects validation_split >= 1");
    }

    cout << "\n=========================================================" << endl;
    cout << " TEST 17: Enum dispatch equivalence (perf refactor)" << endl;
    cout << "=========================================================" << endl;
    {
        vector<pair<string,Activation>> acts = {
            {"sigmoid",Activation::Sigmoid}, {"tanh",Activation::Tanh}, {"ReLU",Activation::ReLU},
            {"linear",Activation::Linear}, {"leaky_ReLU",Activation::LeakyReLU}
        };
        bool same = true, names_ok = true;
        for (const pair<string,Activation>& a : acts) {
            for (double v : {-2.5,-0.4,0.0,0.7,3.1}) {
                if (Neuron::activate(v, a.second) != Neuron::activate(v, a.first)) same = false;
                double out = Neuron::activate(v, a.second);
                if (Neuron::activation_derivative(v, out, a.second) !=
                    Neuron::activation_derivative(v, out, a.first)) same = false;
            }
            if (parse_activation(a.first) != a.second) names_ok = false;
            if (activation_name(a.second) != a.first) names_ok = false;
        }
        check(same, "Enum and string dispatch produce identical values and derivatives");
        check(names_ok, "parse_activation / activation_name round-trip for all activations");

        bool loss_ok = true;
        for (const string& l : {string("mean_squared_error"), string("binary_cross_entropy"),
                                string("categorical_cross_entropy")})
            if (loss_name(parse_loss(l)) != l) loss_ok = false;
        check(loss_ok, "parse_loss / loss_name round-trip for all losses");
        check(activation_name(parse_activation("softmax")) == "softmax", "softmax round-trips by name");

        NeuralNetwork by_enum(3,2,5,1,Activation::Tanh,Activation::Sigmoid,Loss::BinaryCrossEntropy,606);
        NeuralNetwork by_string(3,2,5,1,"tanh","sigmoid","binary_cross_entropy",606);
        check(by_enum.predict({0.2,0.4,0.6}) == by_string.predict({0.2,0.4,0.6}),
              "Enum and string constructors build identical networks");
    }

    cout << "\n=========================================================" << endl;
    cout << " SUMMARY:  " << g_pass << " passed, " << g_fail << " failed" << endl;
    cout << "=========================================================" << endl;
    return g_fail == 0 ? 0 : 1;
}
