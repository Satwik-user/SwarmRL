"""Knowledge sharing over the existing communication-radius links.

When two drones are within Drone.comm_radius (checked via the existing
get_nearby_drones results), confirmed-victim registries are merged. Faint
contacts and explored-cell maps are optional (configurable).
"""


def share_knowledge(drones, knowledge, nearby, mission,
                    share_faint: bool = False, share_maps: bool = True,
                    mesh_simulator=None) -> int:
    id_to_idx = {d.id: i for i, d in enumerate(drones)}
    share_events = 0
    for i, neighbors in enumerate(nearby):
        for nd in neighbors:
            j = id_to_idx.get(nd["drone_id"])
            if j is None or j <= i:
                continue  # each unordered pair once

            # Channel degradation & packet drop simulation if MeshNetworkSimulator is connected
            if mesh_simulator is not None:
                in_range, loss_prob = mesh_simulator.can_communicate(drones[i].pos, drones[j].pos)
                if not in_range or not mesh_simulator.is_packet_delivered(loss_prob):
                    continue

            mission.comm_links += 1
            a, b = knowledge[i], knowledge[j]
            new_a = b.confirmed - a.confirmed
            new_b = a.confirmed - b.confirmed
            if new_a or new_b:
                share_events += 1
                a.confirmed |= new_a
                b.confirmed |= new_b
                for vid in a.confirmed:
                    a.faint.pop(vid, None)
                for vid in b.confirmed:
                    b.faint.pop(vid, None)
            if share_faint:
                for vid, entry in list(b.faint.items()):
                    if vid not in a.faint and vid not in a.confirmed:
                        a.faint[vid] = entry; share_events += 1
                for vid, entry in list(a.faint.items()):
                    if vid not in b.faint and vid not in b.confirmed:
                        b.faint[vid] = entry; share_events += 1
            if share_maps:
                merged = a.visited | b.visited
                # Copies are essential: aliasing would silently turn every
                # future visit into a team-wide update without comms.
                a.visited = merged.copy()
                b.visited = merged.copy()
    mission.comm_share_events += share_events
    return share_events