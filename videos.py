import sys
import gurobipy as gp
from gurobipy import GRB

def solve(dataset_path):
    """
    Lit le fichier de données et résout le problème d'optimisation
    'Streaming Videos' (Hash Code 2017) avec Gurobi.
    """
    print(f"Lecture du dataset : {dataset_path}...")
    
    with open(dataset_path, 'r') as f:
        # Fonction utilitaire pour lire les entiers d'une ligne
        def read_ints():
            line = f.readline()
            while line and not line.strip(): # Ignorer les lignes vides
                line = f.readline()
            if not line:
                return None
            return list(map(int, line.split()))

        # Lecture des paramètres globaux
        header = read_ints()
        V, E, R, C, X = header
        
        # Lecture des tailles des vidéos
        video_sizes = read_ints()
        
        # Lecture des endpoints
        endpoints = []
        for i in range(E):
            e_header = read_ints()
            L_d, K = e_header
            caches = {}
            for _ in range(K):
                c_id, c_lat = read_ints()
                caches[c_id] = c_lat
            endpoints.append({'L_d': L_d, 'caches': caches})
            
        # Lecture des requêtes
        requests = []
        for i in range(R):
            requests.append(read_ints()) # v, e, n

    print("Données lues avec succès.")
    print(f"Paramètres : V={V}, E={E}, R={R}, C={C}, X={X}")

    # Création du modèle
    print("Construction du modèle Gurobi...")
    m = gp.Model("streaming_videos")
    
    # Variables de décision
    # y[c, v] = 1 si la vidéo 'v' est stockée dans le cache 'c'
    y = m.addVars(C, V, vtype=GRB.BINARY, name="y")
    
    # Nous n'avons pas besoin de créer des variables explicites pour chaque arc possible
    # entre cache et endpoint pour chaque requête, car nous cherchons le meilleur gain.
    # Cependant, pour linéariser le choix "servi par le cache c", nous introduisons x[r, c].
    
    # Calcul des gains potentiels (savings)
    # Pour chaque requête r=(v, e, n), si un cache connecté 'c' offre une latence
    # inférieure à celle du data center, c'est un candidat.
    
    obj_expr = 0
    
    # Contrainte de capacité : la somme des tailles des vidéos dans un cache ne doit pas dépasser X
    print("Ajout des contraintes de capacité...")
    m.addConstrs(
        (gp.quicksum(video_sizes[v] * y[c, v] for v in range(V)) <= X for c in range(C)),
        name="Capacity"
    )

    print("Ajout des variables de décision et contraintes de couverture des requêtes...")
    for r_idx, (v, e_id, n) in enumerate(requests):
        e = endpoints[e_id]
        L_d = e['L_d']
        connected_caches = e['caches']
        
        # Liste des variables x_rc pour cette requête spécifique
        r_x_vars = []
        
        for c_id, L_c in connected_caches.items():
            # Si le cache est plus rapide que le Data Center
            if L_c < L_d:
                saving = (L_d - L_c) * n
                # x[r_idx, c_id] = 1 si la requête r est servie par le cache c_id
                x_rc = m.addVar(vtype=GRB.BINARY, name=f"x_{r_idx}_{c_id}")
                
                # Lien logique : on ne peut servir depuis c que si la vidéo y est présente
                # x_rc <= y[c_id, v]
                m.addConstr(x_rc <= y[c_id, v], name=f"Link_{r_idx}_{c_id}")
                
                r_x_vars.append(x_rc)
                obj_expr += saving * x_rc
        
        if r_x_vars:
            # Une requête ne peut être servie que par AU PLUS un cache 
            # (celui qui maximise le gain sera choisi par l'objectif)
            m.addConstr(gp.quicksum(r_x_vars) <= 1, name=f"OneCache_{r_idx}")

    # Objectif : Maximiser le temps total gagné
    m.setObjective(obj_expr, GRB.MAXIMIZE)
    
    # Paramètres du solveur
    # Gap d'optimalité demandé : 0.5%
    m.setParam('MipGap', 0.005) 
    m.setParam('TimeLimit', 600) # Limite de temps de sécurité (10 min)
    
    # Écriture du fichier MPS pour inspection si nécessaire
    print("Génération du fichier videos.mps...")
    m.write("videos.mps")
    
    print("Lancement de l'optimisation...")
    m.optimize()
    
    if m.SolCount > 0:
        print(f"Solution optimale trouvée avec objectif : {m.ObjVal}")
        
        # Génération du fichier de sortie au format requis
        cache_content = {c: [] for c in range(C)}
        for c in range(C):
            for v in range(V):
                if y[c, v].X > 0.5:
                    cache_content[c].append(v)
        
        # On ne liste que les caches utilisés
        used_caches = [c for c in range(C) if cache_content[c]]
        
        with open("videos.out", "w") as out:
            out.write(f"{len(used_caches)}\n")
            for c in used_caches:
                videos_str = " ".join(map(str, cache_content[c]))
                out.write(f"{c} {videos_str}\n")
        
        print("Fichier videos.out généré avec succès.")
    else:
        print("Aucune solution trouvée.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python videos.py <path_to_dataset>")
        sys.exit(1)
    
    solve(sys.argv[1])
