use env_logger::{Builder, Target};
use log::info;
use plotters::{
    prelude::*,
    style::full_palette::{BLUE, RED, WHITE},
};
use rand::{thread_rng, Rng};
use std::{
    fmt::Display,
    ops::Range,
    time::{Duration, Instant},
};

const PLOT_SIZE: (u32, u32) = (1920, 1080);

/// Intervalo inicial para a primeira geração.
const INITIAL_INTERVAL: Range<f64> = -100.0..100.0;

/// Número de indivíduos na população.
const POPULATION_SIZE: usize = 100;

/// Taxa de mutação.
/// **NÃO** é em porcentagem.
const MUTATION_RATE: f64 = 0.001;

/// Intervalo para mutação.
const MUTATION_INTERVAL: Range<f64> = -10.0..10.0;

/// Número máximo de gerações.
const MAX_GENERATIONS: u64 = 100_000;

/// Tolerância permitida para a função de *fitness*.
const FITNESS_TOLERANCE: f64 = 1e-4;

/// Máxima diferença entre dois *best* consecutivos para aplicar o genocídio.
const BEST_DELTA: f64 = 1e-8;

const COUNTER_GENOCIDE: u8 = 5;

enum Selection {
    Elitism,
    Tournament,
}

impl Display for Selection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Selection::Elitism => write!(f, "elitism"),
            Selection::Tournament => write!(f, "tournament"),
        }
    }
}

enum Rearrangement {
    None,
    Genocide,
    RandomPredation,
}

impl Display for Rearrangement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Rearrangement::None => write!(f, ""),
            Rearrangement::Genocide => write!(f, "genocide"),
            Rearrangement::RandomPredation => write!(f, "random_predation"),
        }
    }
}

struct Population {
    selection: Selection,
    rearrangement: Rearrangement,
    run_duration: Option<Duration>,
    range: Range<f64>,
    ind: Vec<f64>,
    generation: u64,
    global_best: Option<f64>,
    best: Option<f64>,
    last_best: Option<f64>,
}

impl Population {
    pub fn new(selection: Selection, rearrangement: Rearrangement) -> Self {
        let mut ind = Vec::with_capacity(POPULATION_SIZE);
        for _ in 0..POPULATION_SIZE {
            let i = individual();
            ind.push(i);
        }
        Self {
            selection,
            rearrangement,
            run_duration: None,
            ind,
            range: INITIAL_INTERVAL,
            generation: 0,
            global_best: None,
            best: None,
            last_best: None,
        }
    }

    /// Retorna o valor do indivíduo presente no `index`.
    pub fn value(&self, index: usize) -> f64 {
        self.ind.get(index).unwrap().clone()
    }

    /// Altera o valor do indivíduo presente no `index` para `v`.
    pub fn set(&mut self, index: usize, v: f64) {
        *self.ind.get_mut(index).unwrap() = v;
    }

    /// Retorna o índice do melhor indivíduo da atual geração.
    pub fn best_index(&self) -> usize {
        let mut current: f64;
        let mut best: f64;
        let mut index: usize = 0;

        best = Population::fitness(self.value(0));

        for i in 0..self.ind.len() {
            current = Population::fitness(self.value(i));
            if current < best {
                best = current;
                index = i;
            }
        }

        index
    }

    pub fn fitness(x: f64) -> f64 {
        function(x).abs()
    }

    fn elitism(&mut self) {
        let best_index = self.best_index();
        let best = self.value(best_index);

        for i in 0..self.ind.len() {
            if i != best_index {
                let mut v = self.value(i);
                v = mutation((v + best) / 2.0);
                self.set(i, v);
            }
        }
        self.generation += 1;
    }

    fn tournament(&mut self) {
        let best_index = self.best_index();
        let best = self.value(best_index);

        let mut dad: f64;
        let mut mom: f64;
        let mut child: f64;

        let mut children = Vec::<f64>::with_capacity(self.ind.len());

        let mut rng = thread_rng();

        for i in 0..self.ind.len() {
            if i != best_index {
                let mut x1 = self.value(rng.gen_range(0..self.ind.len()));
                let mut x2 = self.value(rng.gen_range(0..self.ind.len()));

                dad = if Population::fitness(x1) < Population::fitness(x2) {
                    x1
                } else {
                    x2
                };

                x1 = self.value(rng.gen_range(0..self.ind.len()));
                x2 = self.value(rng.gen_range(0..self.ind.len()));

                mom = if Population::fitness(x1) < Population::fitness(x2) {
                    x1
                } else {
                    x2
                };

                child = mutation((dad + mom) / 2.0);
                children.push(child);
            } else {
                children.push(best);
            }
        }

        for i in 0..self.ind.len() {
            self.set(i, children.get(i).unwrap().clone());
        }

        self.generation += 1;
    }

    fn genocide(&mut self) {
        let m = thread_rng().gen_range(0.1..2.0);
        let start = self.range.start * m;
        let end = self.range.end * m;
        self.range = start..end;
        self.best = None;
        self.last_best = None;
        for i in 0..self.ind.len() {
            self.set(i, thread_rng().gen_range(start..end));
        }
    }

    fn random_predation(&mut self) {
        let mut worst_index = 0;
        let mut worst = Population::fitness(self.value(worst_index));
        for i in 0..self.ind.len() {
            let current = Population::fitness(self.value(i));
            if current > worst {
                worst = current;
                worst_index = i;
            }
        }

        self.set(worst_index, individual());
    }

    pub fn run(&mut self, plot: bool) {
        let now = Instant::now();

        let mut best_data = Vec::<f64>::new();
        let mut aveg_data = Vec::<f64>::new();
        let mut y_max_aveg = 0.0;
        let mut y_max_best = 0.0;
        let mut counter: u8 = 0;

        loop {
            let best = self.value(self.best_index());
            self.last_best = self.best;
            self.best = Some(best);

            match self.global_best {
                Some(global) => {
                    if Population::fitness(best) < Population::fitness(global) {
                        self.global_best = Some(best);
                    }
                }
                None => self.global_best = Some(best),
            }

            if plot {
                let aveg: f64 =
                    self.ind.iter().map(|i| function(*i)).sum::<f64>() / self.ind.len() as f64;
                aveg_data.push(aveg);

                let fitness = Population::fitness(best);
                best_data.push(fitness);

                if fitness > y_max_best {
                    y_max_best = fitness;
                }

                if aveg > y_max_aveg {
                    y_max_aveg = aveg;
                }
            }

            match self.selection {
                Selection::Elitism => self.elitism(),
                Selection::Tournament => self.tournament(),
            }

            match self.rearrangement {
                Rearrangement::None => (),
                Rearrangement::Genocide => {
                    if let Some(best) = self.best {
                        if let Some(last_best) = self.last_best {
                            if (best - last_best).abs() < BEST_DELTA {
                                counter += 1;
                                if counter >= COUNTER_GENOCIDE {
                                    self.genocide();
                                    counter = 0;
                                }
                            } else {
                                counter = 0;
                            }
                        }
                    }
                }
                Rearrangement::RandomPredation => self.random_predation(),
            }

            if Population::fitness(self.global_best.unwrap()) < FITNESS_TOLERANCE
                || self.generation > MAX_GENERATIONS
            {
                break;
            }
        }

        if plot {
            let mut name = "best".to_string();
            let mut caption = "Best by ".to_string();

            match self.selection {
                Selection::Elitism => {
                    name.push_str("_elitism");
                    caption.push_str("Elitism");
                }

                Selection::Tournament => {
                    name.push_str("_tournament");
                    caption.push_str("Tournament")
                }
            }

            match self.rearrangement {
                Rearrangement::None => (),
                Rearrangement::Genocide => name.push_str("_genocide"),
                Rearrangement::RandomPredation => name.push_str("_random_predation"),
            }

            name.push_str(".png");

            plot_data(
                &best_data,
                name.as_str(),
                caption.as_str(),
                0.0..y_max_best,
                BLUE,
            );

            plot_data(
                &aveg_data,
                name.replace("best", "aveg").as_str(),
                caption.replace("Best", "Aveg").as_str(),
                0.0..y_max_aveg,
                RED,
            );
        }

        self.run_duration = Some(now.elapsed());
    }

    pub fn results(&self) {
        let inf = format!(
            "({} ms) - Best by {} ({}): {} | Fitness: {}",
            self.run_duration.unwrap().as_millis(),
            self.selection,
            self.rearrangement,
            self.global_best.unwrap(),
            Population::fitness(self.global_best.unwrap()),
        );

        info!("{}", inf);
    }
}

fn individual() -> f64 {
    thread_rng().gen_range(INITIAL_INTERVAL)
}

fn mutation(i: f64) -> f64 {
    i + thread_rng().gen_range(MUTATION_INTERVAL) * MUTATION_RATE
}

fn plot_data(data: &Vec<f64>, name: &str, caption: &str, y_range: Range<f64>, color: RGBColor) {
    let path = format!("images/{}", name);
    let root_area = BitMapBackend::new(path.as_str(), PLOT_SIZE).into_drawing_area();
    root_area.fill(&WHITE).unwrap();

    let mut ctx = ChartBuilder::on(&root_area)
        .set_label_area_size(LabelAreaPosition::Left, 100)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .set_label_area_size(LabelAreaPosition::Right, 100)
        .caption(caption, ("sans-serif", 40))
        .build_cartesian_2d(0..data.len(), y_range)
        .unwrap();

    ctx.configure_mesh().draw().unwrap();

    ctx.draw_series(LineSeries::new(
        (0..).zip(data.iter()).map(|(x, y)| (x, *y)),
        color,
    ))
    .unwrap();
}

fn main() {
    // mostrar o log (info) no terminal sem precisar ficar setando manualmente a variavel de ambiente
    std::env::set_var("RUST_LOG", "info");

    // logger config
    let mut builder = Builder::from_default_env();
    builder.target(Target::Stdout);
    builder.init();

    {
        let mut pop = Population::new(Selection::Elitism, Rearrangement::None);
        pop.run(true);
        pop.results();
    }

    {
        let mut pop = Population::new(Selection::Tournament, Rearrangement::None);
        pop.run(true);
        pop.results();
    }

    {
        let mut pop = Population::new(Selection::Elitism, Rearrangement::RandomPredation);
        pop.run(true);
        pop.results();
    }

    {
        let mut pop = Population::new(Selection::Elitism, Rearrangement::Genocide);
        pop.run(true);
        pop.results();
    }

    {
        let mut pop = Population::new(Selection::Tournament, Rearrangement::Genocide);
        pop.run(true);
        pop.results();
    }
}

// Função que será avaliada.
fn function(x: f64) -> f64 {
    let y = (x - 478.0) * (x + 4567.0) * (x - 1240.0);
    // let y = x.powi(3) + 97.0 * x.powi(2) + 615.0 * x - 77625.0;

    // let y = 3.0_f64.powf(x) - 9.0_f64.powf(x + 5.0); // x = - 10
    // let y = x * x - 5.0 * x + 6.0;
    // let y = x * x * x - 27.0;
    y
}
