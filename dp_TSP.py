
def tsp(source_node: int, transition_cost: list):
    n_nodes = len(transition_cost)
    memo = {}

    def tsp_(state, current_city):
        '''
        computing optimal cost from current city onwards to last on (also 1st one)
        '''
        visited_frozen = frozenset(state)
        if len(state) == n_nodes:
            memo[visited_frozen, current_city] = transition_cost[current_city][source_node]
            return transition_cost[current_city][source_node]
        
        if (visited_frozen, current_city) in memo:
            return memo[visited_frozen, current_city]

        current_cost = float('inf')
        for city in range(n_nodes):
            if city in state:
                continue
            candidate_cost = transition_cost[current_city][city] + tsp_(state + [city], city)
            if candidate_cost < current_cost:
                current_cost = candidate_cost

        memo[visited_frozen, current_city] = current_cost
        return current_cost
    minimum_cost = tsp_([source_node], source_node)
    return minimum_cost, memo

def get_path(source_node, n_nodes, memo, transition_cost):
    path = [source_node]
    for state in range(2, n_nodes + 1):
        current_city = path[-1]
        min_val = float('inf')
        next_city = -1
        for (path_, city), residual_value in memo.items():

            if ((len(path_) != state) or not (set(path) <= path_) or (city in path)):
                continue
            current_cost = transition_cost[city][current_city] + residual_value
            if current_cost < min_val:
                next_city = city
                min_val = current_cost
        path.append(next_city)
    return path

def main():
    source_node = 5
    transition_cost = [
    #    [0, 10, 15, 20],
    #    [10, 0, 35, 25],
    #    [15, 35, 0, 30],
    #    [20, 25, 30, 0]
    # 5-1-2-0-4-3
    # 5-1: 6, 1-2: 19, 2-0: 29, 0-4: 13, 4-3: 4, 3-5: 5
        [0, 12, 29, 22, 13, 24],
        [12, 0, 19, 3, 25, 6],
        [29, 19, 0, 21, 23, 28],
        [22, 3, 21, 0, 4, 5],
        [13, 25, 23, 4, 0, 16],
        [24, 6, 28, 5, 16, 0],
    ]
    cost, memo = tsp(source_node, transition_cost)
    print(memo)
    path = get_path(source_node, len(transition_cost), memo, transition_cost)
    print(cost)
    print(path)

if __name__ == '__main__':
    main()