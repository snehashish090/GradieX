const features = [
  {
    title: 'Native C++ Performance',
    description:
      'Train and run models close to the metal without Python runtime overhead.',
  },
  {
    title: 'Readable Tensor API',
    description:
      'Compose layers and optimizers with ergonomic C++ primitives and clear data flow.',
  },
  {
    title: 'Production-Ready Inference',
    description:
      'Ship deterministic binaries for edge and server workloads with predictable behavior.',
  },
]

const workflow = [
  'Define a network with familiar layer abstractions.',
  'Train with built-in optimizers and benchmark instantly.',
  'Export and deploy a compact, low-latency model binary.',
]

function App() {
  return (
    <div className="relative overflow-hidden">
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_15%_20%,rgba(15,118,110,0.28),transparent_32%),radial-gradient(circle_at_85%_12%,rgba(234,88,12,0.2),transparent_28%),radial-gradient(circle_at_70%_78%,rgba(190,24,93,0.18),transparent_32%)]" />

      <main className="relative mx-auto max-w-6xl px-6 pb-20 pt-8 sm:px-8 lg:px-10">
        <nav className="mb-14 flex items-center justify-between">
          <div className="inline-flex items-center gap-3">
            <span className="inline-flex h-10 w-10 items-center justify-center rounded-xl bg-teal-500/20 text-lg font-black text-teal-200 ring-1 ring-teal-300/30">
              N
            </span>
            <div>
              <p className="font-display text-lg font-semibold leading-none text-zinc-100">NeuralCpp</p>
              <p className="text-xs uppercase tracking-[0.2em] text-zinc-400">Tooling for fast ML systems</p>
            </div>
          </div>
          <a
            href="#get-started"
            className="rounded-full border border-zinc-700 bg-zinc-900/60 px-4 py-2 text-sm font-medium text-zinc-100 transition hover:border-teal-300/40 hover:bg-zinc-800"
          >
            Documentation
          </a>
        </nav>

        <section className="grid gap-10 lg:grid-cols-[1.1fr_0.9fr] lg:items-end">
          <div className="animate-rise opacity-0 [animation-delay:80ms] [animation-fill-mode:forwards]">
            <p className="mb-4 inline-flex rounded-full border border-teal-300/30 bg-teal-400/10 px-4 py-1 text-xs uppercase tracking-[0.2em] text-teal-100">
              C++ neural network toolkit
            </p>
            <h1 className="font-display text-5xl font-semibold leading-[0.95] text-zinc-100 sm:text-6xl">
              Build neural systems that run as fast as your compiler.
            </h1>
            <p className="mt-6 max-w-xl text-base text-zinc-300 sm:text-lg">
              NeuralCpp gives you modern deep-learning workflows while keeping the reliability and speed of native C++.
            </p>
            <div id="get-started" className="mt-8 flex flex-wrap gap-3">
              <button className="rounded-full bg-teal-400 px-6 py-3 font-semibold text-zinc-950 transition hover:bg-teal-300">
                Start Building
              </button>
              <button className="rounded-full border border-zinc-700 bg-zinc-900 px-6 py-3 font-semibold text-zinc-100 transition hover:border-orange-300/50 hover:text-orange-200">
                View Examples
              </button>
            </div>
          </div>

          <div className="animate-rise opacity-0 [animation-delay:260ms] [animation-fill-mode:forwards]">
            <div className="relative rounded-3xl border border-zinc-800/90 bg-zinc-950/80 p-5 shadow-2xl shadow-black/30 backdrop-blur">
              <div className="mb-4 flex gap-2">
                <span className="h-2.5 w-2.5 rounded-full bg-red-400/70" />
                <span className="h-2.5 w-2.5 rounded-full bg-orange-300/70" />
                <span className="h-2.5 w-2.5 rounded-full bg-teal-300/70" />
              </div>
              <pre className="overflow-x-auto rounded-2xl border border-zinc-800 bg-zinc-900 p-4 text-xs text-zinc-200 sm:text-sm">
                <code>{`Model model;
model.add(Dense(128, Activation::ReLU));
model.add(Dense(64, Activation::ReLU));
model.add(Dense(10, Activation::Softmax));

Trainer trainer(model);
trainer.fit(trainData, 50);

auto output = model.predict(sample);`}</code>
              </pre>
              <div className="mt-5 grid grid-cols-3 gap-2 text-center text-xs text-zinc-300">
                <div className="rounded-xl border border-zinc-800 bg-zinc-900/80 p-2">Low latency</div>
                <div className="rounded-xl border border-zinc-800 bg-zinc-900/80 p-2">Deterministic</div>
                <div className="rounded-xl border border-zinc-800 bg-zinc-900/80 p-2">Portable</div>
              </div>
            </div>
          </div>
        </section>

        <section className="mt-20 grid gap-4 sm:grid-cols-3">
          {features.map((feature, index) => (
            <article
              key={feature.title}
              className="animate-rise rounded-2xl border border-zinc-800 bg-zinc-900/55 p-5 opacity-0 [animation-fill-mode:forwards]"
              style={{ animationDelay: `${320 + index * 120}ms` }}
            >
              <h2 className="font-display text-xl text-zinc-100">{feature.title}</h2>
              <p className="mt-2 text-sm leading-relaxed text-zinc-300">{feature.description}</p>
            </article>
          ))}
        </section>

        <section className="mt-20 grid gap-8 rounded-3xl border border-zinc-800/80 bg-zinc-900/40 p-6 sm:p-8 lg:grid-cols-[1fr_1fr]">
          <div>
            <p className="text-xs uppercase tracking-[0.2em] text-orange-200/85">Workflow</p>
            <h3 className="mt-2 font-display text-3xl text-zinc-100">From idea to binary in a single stack</h3>
            <p className="mt-3 text-zinc-300">
              Keep your architecture, training, and deployment in C++ to reduce context switching and simplify maintenance.
            </p>
          </div>
          <ol className="space-y-4">
            {workflow.map((step, index) => (
              <li key={step} className="rounded-2xl border border-zinc-800 bg-zinc-950/70 p-4">
                <p className="text-xs uppercase tracking-[0.2em] text-zinc-500">Step {index + 1}</p>
                <p className="mt-1 text-sm text-zinc-200">{step}</p>
              </li>
            ))}
          </ol>
        </section>
      </main>
    </div>
  )
}

export default App
