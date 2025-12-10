import math
import random
import time
import matplotlib.pyplot as plt
import re
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import cpu_count


# =====================================================
# LEITURA DA INSTÂNCIA
# =====================================================

def read_tsp_file(path):
    coords = []
    seed = None

    with open(path, "r") as f:
        for line in f:
            s = line.strip()

            if "seed" in s.lower():
                match = re.search(r':\s*(\d+|None)', s, re.IGNORECASE)
                if match and match.group(1) != 'None':
                    seed = int(match.group(1))

            if ':' in s and s[0].isdigit():
                match = re.match(r'(\d+):\s*([\d.]+),\s*([\d.]+)', s)
                if match:
                    x = float(match.group(2))
                    y = float(match.group(3))
                    coords.append((x, y))

    return coords, seed


# =====================================================
# DISTANCE CACHE
# =====================================================

class DistanceCache:
    def __init__(self, coords):
        n = len(coords)
        self.cache = [[0.0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                d = math.dist(coords[i], coords[j])
                self.cache[i][j] = d
                self.cache[j][i] = d
    
    def get(self, i, j):
        return self.cache[i][j]


# =====================================================
# DISTÂNCIAS
# =====================================================

def tour_length(tour, dist_cache):
    total = 0.0
    n = len(tour)
    for i in range(n):
        total += dist_cache.get(tour[i], tour[(i + 1) % n])
    return total


# =====================================================
# 2-OPT LOCAL SEARCH
# =====================================================

def two_opt(tour, dist_cache):
    improved = True
    n = len(tour)

    while improved:
        improved = False
        for i in range(1, n - 2):
            for j in range(i + 1, n):
                if j - i == 1:
                    continue
                a, b = tour[i - 1], tour[i]
                c, d = tour[j - 1], tour[j % n]

                before = dist_cache.get(a, b) + dist_cache.get(c, d)
                after = dist_cache.get(a, c) + dist_cache.get(b, d)

                if after < before:
                    tour[i:j] = reversed(tour[i:j])
                    improved = True

    return tour


# =====================================================
# DOUBLE BRIDGE PERTURBATION
# =====================================================

def double_bridge(tour):
    n = len(tour)
    a = random.randint(1, n // 4)
    b = random.randint(n // 4 + 1, n // 2)
    c = random.randint(n // 2 + 1, 3 * n // 4)
    
    return tour[:a] + tour[c:] + tour[b:c] + tour[a:b]


# =====================================================
# PATH RELINKING
# =====================================================

def path_relinking(tour1, tour2, dist_cache):
    n = len(tour1)
    current = tour1[:]
    best_tour = current[:]
    best_cost = tour_length(current, dist_cache)
    
    for i in range(n):
        if current[i] != tour2[i]:
            target_city = tour2[i]
            current_idx = current.index(target_city)
            
            current[i], current[current_idx] = current[current_idx], current[i]
            
            current = two_opt(current, dist_cache)
            cost = tour_length(current, dist_cache)
            
            if cost < best_cost:
                best_cost = cost
                best_tour = current[:]
    
    return best_tour, best_cost


# =====================================================
# FUNÇÕES WORKER
# =====================================================

def employed_bee_worker(args):
    idx, current_tour, current_cost, dist_cache, n, seed_offset = args
    random.seed(seed_offset + idx)
    
    new_tour = current_tour[:]
    
    if random.random() < 0.7:
        a, b = sorted(random.sample(range(n), 2))
        new_tour[a:b] = reversed(new_tour[a:b])
    else:
        new_tour = double_bridge(new_tour)
    
    new_tour = two_opt(new_tour, dist_cache)
    new_cost = tour_length(new_tour, dist_cache)
    
    improved = new_cost < current_cost
    return (idx, new_tour, new_cost, improved)


def onlooker_bee_worker(args):
    worker_id, food_sources, probabilities, dist_cache, n, seed_offset = args
    random.seed(seed_offset + worker_id + 1000)
    
    idx = random.choices(range(len(food_sources)), probabilities)[0]
    current_tour, current_cost, trial = food_sources[idx]
    
    new_tour = current_tour[:]
    a, b = sorted(random.sample(range(n), 2))
    new_tour[a:b] = reversed(new_tour[a:b])
    
    new_tour = two_opt(new_tour, dist_cache)
    new_cost = tour_length(new_tour, dist_cache)
    
    improved = new_cost < current_cost
    return (idx, new_tour, new_cost, improved)


def path_relinking_worker(args):
    tour1, tour2, dist_cache = args
    return path_relinking(tour1, tour2, dist_cache)


# =====================================================
# ABC PARALELO
# =====================================================

def abc_tsp_parallel(coords, 
                     num_employed=25, 
                     num_onlooker=25, 
                     max_time_seconds=300,
                     limit_stagnation=20,
                     seed=42,
                     n_threads=None,
                     n_processes=None):
    
    if n_threads is None:
        n_threads = cpu_count()
    if n_processes is None:
        n_processes = min(4, cpu_count())
    
    random.seed(seed)
    
    n = len(coords)
    dist_cache = DistanceCache(coords)
    
    print(f"Distance cache: {n}x{n} | Threads: {n_threads} | Processes: {n_processes}")

    food_sources = []
    
    print("Inicializando população...")
    init_start = time.time()
    for _ in range(num_employed):
        tour = list(range(n))
        random.shuffle(tour)
        tour = two_opt(tour, dist_cache)
        food_sources.append([tour, tour_length(tour, dist_cache), 0])
    
    init_time = time.time() - init_start
    print(f"População inicializada em {init_time:.2f}s")

    start_time = time.time()

    best_global = min(food_sources, key=lambda x: x[1])[0]
    best_dist = tour_length(best_global, dist_cache)
    
    print(f"\nABC com limite de tempo: {max_time_seconds}s")
    print(f"Melhor inicial: {best_dist:.4f}")
    print("-" * 60)

    thread_pool = ThreadPoolExecutor(max_workers=n_threads)
    process_pool = ProcessPoolExecutor(max_workers=n_processes)

    iteration = 0
    last_print_time = start_time

    try:
        while True:
            elapsed = time.time() - start_time
            
            if elapsed >= max_time_seconds:
                print(f"\nTempo limite atingido ({max_time_seconds}s)")
                break
            
            iteration += 1
            iter_seed = seed + iteration * 10000
            
            employed_args = [
                (i, fs[0], fs[1], dist_cache, n, iter_seed)
                for i, fs in enumerate(food_sources)
            ]
            
            employed_results = list(thread_pool.map(employed_bee_worker, employed_args))
            
            for idx, new_tour, new_cost, improved in employed_results:
                if improved:
                    food_sources[idx] = [new_tour, new_cost, 0]
                else:
                    food_sources[idx][2] += 1

            costs = [fs[1] for fs in food_sources]
            probs = [(max(costs) - c + 1) for c in costs]
            s = sum(probs)
            probs = [p / s for p in probs]
            
            onlooker_args = [
                (i, food_sources, probs, dist_cache, n, iter_seed)
                for i in range(num_onlooker)
            ]
            
            onlooker_results = list(thread_pool.map(onlooker_bee_worker, onlooker_args))
            
            for idx, new_tour, new_cost, improved in onlooker_results:
                if improved and new_cost < food_sources[idx][1]:
                    food_sources[idx] = [new_tour, new_cost, 0]
                else:
                    food_sources[idx][2] += 1

            scouts_activated = 0
            for i in range(num_employed):
                if food_sources[i][2] >= limit_stagnation:
                    new_tour = list(range(n))
                    random.shuffle(new_tour)
                    new_tour = double_bridge(new_tour)
                    new_tour = two_opt(new_tour, dist_cache)
                    new_cost = tour_length(new_tour, dist_cache)
                    food_sources[i] = [new_tour, new_cost, 0]
                    scouts_activated += 1

            if iteration % 20 == 0:
                sorted_sources = sorted(food_sources, key=lambda x: x[1])
                top_4 = [fs[0] for fs in sorted_sources[:4]]
                
                pr_pairs = [
                    (top_4[0], top_4[1], dist_cache),
                    (top_4[0], top_4[2], dist_cache),
                    (top_4[1], top_4[2], dist_cache),
                    (top_4[0], top_4[3], dist_cache),
                ]
                
                pr_results = list(process_pool.map(path_relinking_worker, pr_pairs))
                
                best_pr_tour, best_pr_cost = min(pr_results, key=lambda x: x[1])
                
                if best_pr_cost < sorted_sources[0][1]:
                    worst_idx = max(range(num_employed), key=lambda i: food_sources[i][1])
                    food_sources[worst_idx] = [best_pr_tour, best_pr_cost, 0]
                    print(f"[PATH RELINKING] Iter {iteration} | Solução: {best_pr_cost:.4f}")

            current_best = min(food_sources, key=lambda x: x[1])
            if current_best[1] < best_dist:
                improvement = best_dist - current_best[1]
                improvement_pct = (improvement / best_dist) * 100
                best_dist = current_best[1]
                best_global = current_best[0]
                elapsed = time.time() - start_time
                print(f"[MELHORIA] Iter {iteration:3d} | Dist: {best_dist:.4f} | +{improvement:.4f} ({improvement_pct:.2f}%) | {elapsed:.2f}s")

            current_time = time.time()
            if current_time - last_print_time >= 10:
                elapsed = time.time() - start_time
                avg_cost = sum(fs[1] for fs in food_sources) / num_employed
                remaining = max_time_seconds - elapsed
                print(f"Iter {iteration:3d} | Best: {best_dist:.4f} | Avg: {avg_cost:.4f} | Scouts: {scouts_activated} | {elapsed:.1f}s | Restante: {remaining:.1f}s")
                last_print_time = current_time
    
    finally:
        thread_pool.shutdown(wait=True)
        process_pool.shutdown(wait=True)
    
    total_time = time.time() - start_time
    print(f"\nIterações executadas: {iteration}")
    print(f"Tempo total: {total_time:.2f}s")
    
    return best_global, best_dist, total_time


# =====================================================
# PLOT
# =====================================================

def plot_tour(coords, tour, save_path="tour_plot.png"):
    xs = [coords[i][0] for i in tour] + [coords[tour[0]][0]]
    ys = [coords[i][1] for i in tour] + [coords[tour[0]][1]]

    plt.figure(figsize=(9, 9))
    plt.plot(xs, ys, "-b", linewidth=1)
    plt.scatter(xs, ys, c="red", s=30)

    dist_cache = DistanceCache(coords)
    title_text = f"Melhor rota: {tour_length(tour, dist_cache):.2f}"
    plt.title(title_text)
    plt.tight_layout()
    
    # Salvar gráfico
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Gráfico salvo: {save_path}")
    
    plt.show()


# =====================================================
# SALVAR RESULTADOS
# =====================================================

def save_results(path, best_dist, total_time, n):
    with open(path, "w") as f:
        f.write("## ABC + 2-OPT + PATH RELINKING\n")
        f.write(f"# Instancia: {n} cidades\n")
        f.write(f"# Melhor distancia: {best_dist:.4f}\n")
        f.write(f"# Tempo: {total_time:.4f}s\n")
    print(f"Resultados salvos: {path}")


# =====================================================
# MAIN
# =====================================================

def main():
    coords, seed = read_tsp_file("sample.txt")
    if seed is None:
        seed = 42

    print(f"Instancia: {len(coords)} cidades | Seed: {seed}")

    max_time = 300
    best_tour, best_dist, elapsed = abc_tsp_parallel(coords, max_time_seconds=max_time, seed=seed)

    print(f"\n{'='*60}")
    print(f"RESULTADO FINAL")
    print(f"{'='*60}")
    print(f"Melhor distancia: {best_dist:.2f}")
    print(f"Tempo de execucao: {elapsed:.2f}s")
    print(f"{'='*60}")

    save_results("results.txt", best_dist, elapsed, len(coords))
    
    print("\nGerando visualizacao...")
    plot_tour(coords, best_tour)


if __name__ == "__main__":
    main()
