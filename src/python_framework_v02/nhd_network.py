from collections import defaultdict, Counter, deque
from itertools import chain
from functools import reduce, partial
from collections.abc import Iterable


def nodes(N):
    yield from N.keys() | (v for v in chain.from_iterable(N.values()) if v not in N)


def edges(N):
    for i, v in N.items():
        for j in v:
            yield (i, j)


def in_degrees(N):
    """
    Compute indegree of nodes in N.

    Args:
        N (dict): Network

    Returns:

    """
    degs = Counter(chain.from_iterable(N.values()))
    degs.update(dict.fromkeys(headwaters(N), 0))
    return degs


def out_degrees(N):
    """
    Compute outdegree of nodes in N

    Args:
        N (dict): Network

    Returns:

    """
    return in_degrees(reverse_network(N))


def extract_connections(rows, target_col, terminal_code=0):
    """Extract connection network from dataframe.

    Arguments:
        rows (DataFrame): Dataframe indexed by key_col.
        key_col (str): Source of each edge
        target_col (str): Target of edge

    Returns:
        (dict)
    """
    network = {}
    for src, dst in rows[target_col].items():
        if src not in network:
            network[src] = []

        if dst > 0:
            network[src].append(dst)
    return network


def extract_waterbodies(rows, target_col, waterbody_null=-9999):
    """Extract waterbody mapping from dataframe.
    """
    return rows.loc[rows[target_col] != waterbody_null, target_col].to_dict()


def reverse_surjective_mapping(d):
    rd = defaultdict(list)
    for src, dst in d.items():
        rd[dst].append(src)
    rd.default_factory = None
    return rd


def reverse_network(N):
    rg = defaultdict(list)
    for src, dst in N.items():
        rg[src]
        for n in dst:
            rg[n].append(src)
    rg.default_factory = None
    return rg


def junctions(N):
    c = Counter(chain.from_iterable(N.values()))
    return {k for k, v in c.items() if v > 1}


def headwaters(N):
    yield from N.keys() - chain.from_iterable(N.values())


def tailwaters(N):
    yield from chain.from_iterable(N.values()) - N.keys()
    yield from (m for m, n in N.items() if not n)


def reachable(N, sources=None, targets=None):
    """
    Return nodes reachable from sources.
    Args:
        N:
        sources (iterable): If None, source nodes are used.
        targets (iterable): Target nodes to stop searching.

    Returns:
    """
    if sources is None:
        sources = headwaters(N)

    rv = {}
    if targets is None:
        for h in sources:
            reach = set()
            Q = deque([h])
            while Q:
                x = Q.popleft()
                reach.add(x)
                Q.extend(N.get(x, ()))
            rv[h] = reach
    else:
        targets = set(targets)

        for h in sources:
            reach = set()
            Q = deque([h])
            while Q:
                x = Q.popleft()
                reach.add(x)
                if x not in targets:
                    Q.extend(N.get(x, ()))
            rv[h] = reach
    return rv


def reachable_network(N, sources=None, targets=None, check_disjoint=True):
    """
    Return subnetworks generated by reach
    Args:
        N:
        sources:

    Returns:

    """
    reached = reachable(N, sources=sources, targets=targets)
    if check_disjoint and reduce(set.intersection, reached.values()):
        raise ValueError("Networks not disjoint")

    rv = {}
    for k, n in reached.items():
        rv[k] = {m: N.get(m, []) for m in n}
    return rv


def split_at_junction(network, path, node):
    return len(network[node]) == 1


def split_at_waterbodies_and_junctions(waterbody_nodes, network, path, node):
    if path[-1] in waterbody_nodes:
        return node in waterbody_nodes
    else:
        return len(network[node]) == 1
    #return node not in waterbody_nodes and len(network[node]) == 1



def dfs_decomposition(N, path_func, source_nodes=None):
    """
    Decompose N into a list of simple segments.
    The order of these segments are suitable to be parallelized as we guarantee that for any segment,
    the predecessor segments appear before it in the list.

    This is accomplished by a depth first search on the reversed graph and
    finding the path from node to its nearest junction.

    Arguments:
        N (Dict[obj: List[obj]]): The graph

    Returns:
        [List]: List of paths to be processed in order.
    """
    if source_nodes is None:
        source_nodes = headwaters(N)

    paths = []
    visited = set()
    for h in source_nodes:
        stack = [(h, iter(N[h]))]
        while stack:
            node, children = stack[-1]
            try:
                child = next(children)
                if child not in visited:
                    # Check to see if we are at a leaf
                    if child in N:
                        stack.append((child, iter(N[child])))
                    visited.add(child)
            except StopIteration:
                node, _ = stack.pop()
                path = [node]

                for n, _ in reversed(stack):
                    if path_func(path, n):
                        path.append(n)
                    else:
                        break
                paths.append(path)
                if len(path) > 1:
                    # Only pop ancestor nodes that were added by path_func.
                    del stack[-(len(path) - 1) :]

    return paths


def segment_deps(segments, connections):
    """Build a dependency graph of segments

    Arguments:
        segments (list): List of paths
        connections {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    # index segements
    index = {d[0]: i for i, d in enumerate(segments)}
    deps = defaultdict(list)
    for i, s in enumerate(segments):
        cand = s[-1]
        if cand in connections:
            if connections[cand]:
                # There is a node downstream
                deps[i].append(index[connections[cand][0]])
    return deps


def kahn_toposort(N):
    degrees = in_degrees(N)
    zero_degree = set(k for k, v in degrees.items() if v == 0)

    _deg_pop = zero_degree.pop
    _deg_add = zero_degree.add
    _network_get = N.get
    while zero_degree:
        n = _deg_pop()
        for j in _network_get(n, ()):
            degrees[j] = c = degrees[j] - 1
            if c == 0:
                _deg_add(j)
        yield n

    try:
        next(degrees.elements())
        raise Exception("Cycle exists!")
    except StopIteration:
        pass


def kahn_toposort_edges(N):
    sorted_nodes = kahn_toposort(N)
    for n in sorted_nodes:
        for m in N.get(n, ()):
            yield (n, m)


def reservoir_shore(connections, waterbody_nodes):
    wbody_set = set(waterbody_nodes)
    not_in = lambda x: x not in wbody_set

    shore = set()
    for node in wbody_set:
        shore.update(filter(not_in, connections[node]))
    return sorted(shore)


def reservoir_boundary(connections, waterbodies, n):
    if n not in waterbodies and n in connections:
        return any(x in waterbodies for x in connections[n])
    return False


def separate_waterbodies(connections, waterbodies):
    waterbody_nodes = {}
    for wb, nodes in reverse_surjective_mapping(waterbodies).items():
        waterbody_nodes[wb] = net = {}
        for n in nodes:
            if n in connections:
                net[n] = list(filter(waterbodies.__contains__, connections[n]))
    return waterbody_nodes


def replace_waterbodies_connections(connections, waterbodies):
    """
    Use a single node to represent waterbodies. The node id is the
    waterbody id.

    This returns a new copy of connections with transformation
    """
    new_conn = {}
    waterbody_nets = separate_waterbodies(connections, waterbodies)

    reversed_conns = reverse_network(connections)
    for n in connections:
        if n in waterbodies:
            wbody_code = waterbodies[n]
            if wbody_code in new_conn:
                continue

            # get all nodes from waterbody
            wbody_nodes = [k for k, v in waterbodies.items() if v == wbody_code]
            outgoing = reservoir_shore(connections, wbody_nodes)
            new_conn[wbody_code] = outgoing

        elif reservoir_boundary(connections, waterbodies, n):
            # one of the children of n is a member of a waterbody
            # replace that child with waterbody code.
            new_conn[n] = []

            for child in connections[n]:
                if child in waterbodies:
                    new_conn[n].append(waterbodies[child])
                else:
                    new_conn[n].append(child)
        else:
            # copy to new network unchanged
            new_conn[n] = connections[n]
    return new_conn
