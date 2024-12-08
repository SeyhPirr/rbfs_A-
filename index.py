import math
import copy
import time
import heapq

puzzle = [
    [1, 2, 3],
    [4, 0, 5],
    [6, 7, 8]
]

goal_state = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 0]
]

#0 i yani boslugu yukari kaydir.
def move_up(puzzle, zero_pos):
    x, y = zero_pos
    if x > 0:  
        new_puzzle = copy.deepcopy(puzzle)
        new_puzzle[x][y], new_puzzle[x - 1][y] = new_puzzle[x - 1][y], new_puzzle[x][y]  
        return new_puzzle, (x - 1, y)
    return None, zero_pos  

# 0 i asagi kaydir.
def move_down(puzzle, zero_pos):
    x, y = zero_pos
    if x < 2:  
        new_puzzle = copy.deepcopy(puzzle)
        new_puzzle[x][y], new_puzzle[x + 1][y] = new_puzzle[x + 1][y], new_puzzle[x][y]  
        return new_puzzle, (x + 1, y)
    return None, zero_pos

# 0 i sola kaydir.
def move_left(puzzle, zero_pos):
    x, y = zero_pos
    if y > 0:  
        new_puzzle = copy.deepcopy(puzzle)
        new_puzzle[x][y], new_puzzle[x][y - 1] = new_puzzle[x][y - 1], new_puzzle[x][y]  
        return new_puzzle, (x, y - 1)
    return None, zero_pos

# 0 i saga kaydir
def move_right(puzzle, zero_pos):
    x, y = zero_pos
    if y < 2: 
        new_puzzle = copy.deepcopy(puzzle)
        new_puzzle[x][y], new_puzzle[x][y + 1] = new_puzzle[x][y + 1], new_puzzle[x][y]  
        return new_puzzle, (x, y + 1)
    return None, zero_pos


#yardimci fonksiyon
def find_goal_position(value):
    for i in range(3):
        for j in range(3):
            if goal_state[i][j] == value:
                return (i, j)
    return None

#suanki puzzle durumundan mumkun olan nodelari cikaran fonksiyon
def generate_nodes(puzzle, zero_pos):
    nodes = []
    moves = [
        move_up(puzzle, zero_pos),
        move_down(puzzle, zero_pos),
        move_left(puzzle, zero_pos),
        move_right(puzzle, zero_pos)
    ]
    for new_puzzle, new_zero_pos in moves:
        if new_puzzle:
            nodes.append((new_puzzle, new_zero_pos))
    return nodes

#heuristic fonksiyonlar
def manhattan_distance(puzzle):
    total_distance = 0
    for i in range(3):
        for j in range(3):
            if puzzle[i][j] != 0:
                goal_x, goal_y = find_goal_position(puzzle[i][j])
                total_distance += abs(i - goal_x) + abs(j - goal_y)
    return total_distance

def misplaced_tiles(puzzle):
    misplaced_count = 0
    for i in range(3):
        for j in range(3):
            if puzzle[i][j] != 0 and puzzle[i][j] != goal_state[i][j]:
                misplaced_count += 1
    return misplaced_count

def sqrt_manhattan_distance(puzzle):
    total_distance = 0
    for i in range(3):
        for j in range(3):
            if puzzle[i][j] != 0:
                goal_x, goal_y = find_goal_position(puzzle[i][j])
                distance = abs(i - goal_x) + abs(j - goal_y)
                total_distance += math.sqrt(distance)
    return total_distance

def maximum_heuristic(puzzle):
    max_distance = 0
    for i in range(3):
        for j in range(3):
            if puzzle[i][j] != 0:
                goal_x, goal_y = find_goal_position(puzzle[i][j])
                distance = abs(i - goal_x) + abs(j - goal_y)
                max_distance = max(max_distance, distance)
    return max_distance

#######
def evaluate_node_heuristics(nodes):
    results = []
    for index, (node, zero_pos) in enumerate(nodes):
        heuristics = {
            "Manhattan Distance": manhattan_distance(node),
            "Misplaced Tiles": misplaced_tiles(node),
            "Square Root of Manhattan Distance": sqrt_manhattan_distance(node),
            "Maximum Heuristic": maximum_heuristic(node)
        }
        results.append((node, heuristics))
        print(f"Node {index + 1}:")
        for row in node:
            print(row)
        print("Heuristics:", heuristics)
        print()
    return results

zero_pos = (1, 1)



def a_star(puzzle, heuristic_func):
    start = puzzle  # Başlangıç durumunu al
    zero_pos = (1, 1)  # Boş kutunun (0'ın) başlangıç konumu
    frontier = []  # Öncelik kuyruğunu başlat (açık liste)
    # İlk durum için (heuristic değer, maliyet, durum, boş konum, yol) eklenir
    heapq.heappush(frontier, (heuristic_func(start), 0, start, zero_pos, []))  
    explored = set()  # Ziyaret edilen durumları saklayan küme
    nodes_explored = 0  # Keşfedilen düğüm sayacı

    while frontier:  # Öncelik kuyruğu boş olmadığı sürece döngüye devam et
        # Kuyruktan en düşük toplam maliyetli durumu al
        _, cost, current, zero_pos, path = heapq.heappop(frontier)
        nodes_explored += 1  # Keşfedilen düğüm sayısını bir artır

        # Eğer mevcut durum hedef durumsa, çözüm yolunu ve düğüm sayısını döndür
        if current == goal_state:
            return path, nodes_explored  

        # Mevcut durumu ziyaret edilenlere ekle
        explored.add(tuple(map(tuple, current)))

        # Komşu durumları üret ve her biri için işlem yap
        for neighbor, new_zero_pos in generate_nodes(current, zero_pos):
            # Eğer komşu durum ziyaret edilmediyse
            if tuple(map(tuple, neighbor)) not in explored:
                new_cost = cost + 1  # Geçerli maliyeti bir artır
                total_cost = new_cost + heuristic_func(neighbor)  # Toplam maliyeti hesapla
                # Komşuyu öncelik kuyruğuna ekle
                heapq.heappush(frontier, (total_cost, new_cost, neighbor, new_zero_pos, path + [neighbor]))

    # Eğer çözüm bulunamazsa, None ve keşfedilen düğüm sayısını döndür
    return None, nodes_explored
heuristics = {
    "Manhattan Distance": manhattan_distance,
    "Misplaced Tiles": misplaced_tiles,
    "Square Root of Manhattan Distance": sqrt_manhattan_distance,
    "Maximum Heuristic": maximum_heuristic
}

for name, heuristic in heuristics.items():
    print(f" A*  {name} Heuristic")
    path, explored = a_star(puzzle, heuristic)
    print(f"Cozum yolu uzunlugu: {len(path) if path else 'No solution'}")
    print(f"kesfedilen nodelar: {explored}")
    print("-" * 40)

### rbfs


def rbfs(puzzle, goal, zero_pos, depth, f_limit, g, heuristics):
    if puzzle == goal:
        return True, []

    f_value = g + heuristics(puzzle)

    if f_value > f_limit:
        return False, f_value

    nodes = generate_nodes(puzzle, zero_pos)
    successors = []

    for new_puzzle, new_zero_pos in nodes:
        new_g = g + 1
        successors.append((new_puzzle, new_zero_pos, new_g, new_g + heuristics(new_puzzle)))

    if not successors:
        return False, float('inf')

    successors.sort(key=lambda x: x[3])

    while True:
        best_successor = successors[0]
        best_puzzle, best_zero_pos, best_g, best_f = best_successor

        if best_f > f_limit:
            return False, best_f

        alternative_f = successors[1][3] if len(successors) > 1 else float('inf')

        success, result = rbfs(best_puzzle, goal, best_zero_pos, depth + 1, min(f_limit, alternative_f), best_g, heuristics)

        if success:
            return True, [best_puzzle] + result

        successors[0] = (best_puzzle, best_zero_pos, best_g, result)
        successors.sort(key=lambda x: x[3])


def run_rbfs(puzzle, heuristics):
    zero_pos = (1, 1)  
    f_limit = heuristics(puzzle)
    g = 0 

    found, path = rbfs(puzzle, goal_state, zero_pos, 0, f_limit, g, heuristics)
    print(found)
    if found:
        return path
    else:
        return [] 

heuristics = {
    "Manhattan Distance": manhattan_distance,
    "Misplaced Tiles": misplaced_tiles,
    "Square Root of Manhattan Distance": sqrt_manhattan_distance,
    "Maximum Heuristic": maximum_heuristic
}

# RBFS'yi çalıştır
for name, heuristic in heuristics.items():
    print(f" RBFS  {name} Heuristic")
    path = run_rbfs(puzzle, heuristic)
    print(f"Cozum yol uzunlugu: {len(path) if path else 'cozum yok'}")
    print("-" * 40)
