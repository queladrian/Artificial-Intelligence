import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import heapq

########################### INFORMATION ABOUT LOCATIONS (NODES AND EDGES) ##############################################

initial_state = (0.405089,-78.168223)
goal_state = (0.405001, -78.174156)

G = ox.graph.graph_from_point(
    initial_state,
    dist=1000,
    network_type="walk", # I walk to Library from my home located on San Vicente
    simplify=True
)

start = ox.distance.nearest_nodes(
    G,
    X=initial_state[1],
    Y=initial_state[0]
)

goal = ox.distance.nearest_nodes(
    G,
    X=goal_state[1],
    Y=goal_state[0]
)

print("Nodes:", len(G.nodes))
print("Edges:", len(G.edges))


######################## DEFINITION OF HEURISTIC FUNCTION #########################

def heuristic(node, goal):
    lat1 = G.nodes[node]["y"]
    lon1 = G.nodes[node]["x"]

    lat2 = G.nodes[goal]["y"]
    lon2 = G.nodes[goal]["x"]

    return ox.distance.great_circle(
        lat1, lon1, lat2, lon2
    )


########################## A* ALGORITHM #####################################

def astar(G, start, goal, custom=False):

    frontier = []

    heapq.heappush(
        frontier,
        (0, start)
    )

    came_from = {start: None}
    cost_so_far = {start: 0}

    while frontier:

        _, current = heapq.heappop(frontier)

        if current == goal:
            break

        for neighbor in G.neighbors(current):

            
            edges = G.get_edge_data(current, neighbor)

            edge = min(
                edges.values(),
                key=lambda x: x.get("length", 1)
            )

            distance = edge.get("length", 1)

            

            if custom: # I chose two parameters to calculate the custom cost: surface and highway. 

                surface = edge.get("surface", "unknown")
                highway = edge.get("highway", "unknown")

                if surface in ["gravel", "dirt", "ground", "unpaved"]:
                    surface_cost = 1.3
                else:
                    surface_cost = 1.0

                if highway == "primary":
                    traffic_cost = 2.0
                elif highway == "secondary":
                    traffic_cost = 1.5
                elif highway == "tertiary":
                    traffic_cost = 1.2
                else:
                    traffic_cost = 1.0

                edge_cost = distance * surface_cost * traffic_cost 

            else:
                edge_cost = distance

            new_cost = cost_so_far[current] + edge_cost

            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:

                cost_so_far[neighbor] = new_cost

                priority = (
                    new_cost
                    + heuristic(neighbor, goal)
                )

                heapq.heappush(
                    frontier,
                    (priority, neighbor)
                )

                came_from[neighbor] = current


    route = []

    current = goal

    while current is not None:
        route.append(current)
        current = came_from[current]

    route.reverse()

    return route, cost_so_far[goal]

####################################################################################

shortest_route, shortest_cost = astar(
    G,
    start,
    goal
)

print("Information about the shortest route:")
print("Number of nodes:", len(shortest_route))
print("Distance:", round(shortest_cost, 1), "m")


############################### CUSTOM COST (QUIET ROUTE) ####################################

quiet_route, quiet_cost = astar(
    G,
    start,
    goal,
    custom=True
)

print("Information about the cleanest route:")
print("Number of nodes:", len(quiet_route))
print("Custom cost is:", round(quiet_cost, 1))

########################## COMPARISON WITH NETWORKX ALGORITHM ####################################

check_route = nx.astar_path(
    G,
    start,
    goal,
    weight="length"
)

check_distance = nx.path_weight(
    G,
    check_route,
    weight="length"
)

print("Verification with NetworkX A* algorithm:")
print("NetworkX distance is:", round(check_distance, 1), "m")

########################### PLOT ROUTES ####################################

ox.plot_graph_route(
    G,
    quiet_route,
    route_color="red",
    route_linewidth=4,
    node_size=0,
    bgcolor="white"
)

plt.show()

