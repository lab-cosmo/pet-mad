##################################
 Dataset Exploration with PET-MAD
##################################

This tutorial covers how to use PET-MAD's featurizer for dataset
visualization and exploration.

**************
 Introduction
**************

PET-MAD includes tools for exploring and visualizing datasets using the
learned representations from the model. The ``PETMADFeaturizer`` can:

-  Extract high-dimensional features from atomic structures
-  Reduce dimensionality for visualization
-  Identify structural and chemical patterns
-  Create interactive visualizations with chemiscope

The featurizer uses the last-layer features from PET-MAD combined with
sketch-map dimensionality reduction to create meaningful 2D and 3D
projections.

*************
 Basic Usage
*************

Setting up the Featurizer
=========================

.. code:: python

   from pet_mad.explore import PETMADFeaturizer
   import ase.io
   import numpy as np

   # Initialize the featurizer
   featurizer = PETMADFeaturizer(version="latest", device="cpu")

   # Load some structures (example with built-in ASE structures)
   from ase.build import bulk, molecule

   structures = [
       bulk("Si", cubic=True, a=5.43, crystalstructure="diamond"),
       bulk("C", cubic=True, a=3.55, crystalstructure="diamond"),
       bulk("Ge", cubic=True, a=5.66, crystalstructure="diamond"),
       molecule("H2O"),
       molecule("CO2"),
       molecule("CH4"),
   ]

Basic Feature Extraction
========================

.. code:: python

   # Extract features
   features = featurizer(structures, None)

   print(f"Feature matrix shape: {features.shape}")
   print(f"Number of structures: {len(structures)}")
   print(f"Feature dimensionality: {features.shape[1]}")

*******************************
 Working with Trajectory Files
*******************************

Loading Trajectory Data
=======================

.. code:: python

   # Load structures from trajectory file
   # frames = ase.io.read("trajectory.xyz", ":")

   # For demonstration, create a synthetic trajectory
   import numpy as np


   def create_demo_trajectory():
       """Create a demonstration trajectory with different Si structures"""
       frames = []

       # Different lattice parameters
       for a in np.linspace(5.2, 5.8, 10):
           atoms = bulk("Si", cubic=True, a=a, crystalstructure="diamond")
           frames.append(atoms)

       # Different crystal structures
       frames.append(bulk("Si", cubic=True, a=5.43, crystalstructure="fcc"))
       frames.append(bulk("Si", cubic=True, a=5.43, crystalstructure="bcc"))

       return frames


   trajectory_frames = create_demo_trajectory()

   # Extract features from trajectory
   trajectory_features = featurizer(trajectory_frames, None)
   print(f"Trajectory features shape: {trajectory_features.shape}")

Batch Processing Large Datasets
===============================

.. code:: python

   def process_large_dataset(structures, batch_size=100):
       """Process large datasets in batches"""

       all_features = []

       for i in range(0, len(structures), batch_size):
           batch = structures[i : i + batch_size]
           batch_features = featurizer(batch, None)
           all_features.append(batch_features)

           print(
               f"Processed batch {i//batch_size + 1}/{(len(structures)-1)//batch_size + 1}"
           )

       # Combine all features
       combined_features = np.vstack(all_features)
       return combined_features


   # Example with larger dataset
   large_structures = []
   for i in range(50):
       # Create variations
       a = 5.43 + np.random.normal(0, 0.1)
       atoms = bulk("Si", cubic=True, a=a, crystalstructure="diamond")
       large_structures.append(atoms)

   large_features = process_large_dataset(large_structures, batch_size=10)

****************************
 Visualization and Analysis
****************************

Basic Feature Analysis
======================

.. code:: python

   import matplotlib.pyplot as plt
   from sklearn.decomposition import PCA
   from sklearn.manifold import TSNE

   # The featurizer already provides low-dimensional projections
   # but we can also apply additional dimensionality reduction

   # PCA analysis
   pca = PCA(n_components=2)
   features_pca = pca.fit_transform(features)

   # t-SNE analysis (for comparison)
   tsne = TSNE(n_components=2, random_state=42)
   features_tsne = tsne.fit_transform(features)

   # Create labels for visualization
   labels = ["Si diamond", "C diamond", "Ge diamond", "H2O", "CO2", "CH4"]

   # Plot results
   fig, axes = plt.subplots(1, 3, figsize=(15, 5))

   # PET-MAD features (already low-dimensional)
   axes[0].scatter(features[:, 0], features[:, 1], s=100, alpha=0.7)
   for i, label in enumerate(labels):
       axes[0].annotate(
           label,
           (features[i, 0], features[i, 1]),
           xytext=(5, 5),
           textcoords="offset points",
       )
   axes[0].set_title("PET-MAD Features")
   axes[0].set_xlabel("Feature 1")
   axes[0].set_ylabel("Feature 2")

   # PCA
   axes[1].scatter(features_pca[:, 0], features_pca[:, 1], s=100, alpha=0.7)
   for i, label in enumerate(labels):
       axes[1].annotate(
           label,
           (features_pca[i, 0], features_pca[i, 1]),
           xytext=(5, 5),
           textcoords="offset points",
       )
   axes[1].set_title("PCA")
   axes[1].set_xlabel("PC1")
   axes[1].set_ylabel("PC2")

   # t-SNE
   axes[2].scatter(features_tsne[:, 0], features_tsne[:, 1], s=100, alpha=0.7)
   for i, label in enumerate(labels):
       axes[2].annotate(
           label,
           (features_tsne[i, 0], features_tsne[i, 1]),
           xytext=(5, 5),
           textcoords="offset points",
       )
   axes[2].set_title("t-SNE")
   axes[2].set_xlabel("t-SNE 1")
   axes[2].set_ylabel("t-SNE 2")

   plt.tight_layout()
   plt.show()

Clustering Analysis
===================

.. code:: python

   from sklearn.cluster import KMeans, DBSCAN
   from sklearn.metrics import silhouette_score

   # K-means clustering
   n_clusters = 3
   kmeans = KMeans(n_clusters=n_clusters, random_state=42)
   cluster_labels = kmeans.fit_predict(features)

   # DBSCAN clustering
   dbscan = DBSCAN(eps=0.5, min_samples=2)
   dbscan_labels = dbscan.fit_predict(features)

   # Evaluate clustering
   if len(set(cluster_labels)) > 1:
       kmeans_score = silhouette_score(features, cluster_labels)
       print(f"K-means silhouette score: {kmeans_score:.3f}")

   if len(set(dbscan_labels)) > 1 and -1 not in dbscan_labels:
       dbscan_score = silhouette_score(features, dbscan_labels)
       print(f"DBSCAN silhouette score: {dbscan_score:.3f}")

   # Visualize clusters
   fig, axes = plt.subplots(1, 2, figsize=(12, 5))

   # K-means
   scatter = axes[0].scatter(
       features[:, 0], features[:, 1], c=cluster_labels, s=100, alpha=0.7, cmap="viridis"
   )
   axes[0].set_title("K-means Clustering")
   axes[0].set_xlabel("Feature 1")
   axes[0].set_ylabel("Feature 2")
   plt.colorbar(scatter, ax=axes[0])

   # DBSCAN
   scatter = axes[1].scatter(
       features[:, 0], features[:, 1], c=dbscan_labels, s=100, alpha=0.7, cmap="viridis"
   )
   axes[1].set_title("DBSCAN Clustering")
   axes[1].set_xlabel("Feature 1")
   axes[1].set_ylabel("Feature 2")
   plt.colorbar(scatter, ax=axes[1])

   plt.tight_layout()
   plt.show()

*******************************************
 Interactive Visualization with Chemiscope
*******************************************

Basic Chemiscope Integration
============================

.. code:: python

   # Note: This requires chemiscope to be installed
   # pip install chemiscope

   try:
       import chemiscope

       # Create interactive visualization
       # This works best in Jupyter notebooks
       chemiscope.explore(structures, featurize=featurizer)

   except ImportError:
       print("Chemiscope not available. Install with: pip install chemiscope")
       print("Interactive visualization requires Jupyter notebook environment")

Custom Properties for Visualization
===================================

.. code:: python

   # Add custom properties for enhanced visualization
   def add_custom_properties(structures):
       """Add custom properties for visualization"""

       properties = []

       for atoms in structures:
           # Calculate basic properties
           n_atoms = len(atoms)
           volume = atoms.get_volume()
           density = len(atoms) / volume

           # Get chemical composition
           symbols = atoms.get_chemical_symbols()
           unique_elements = set(symbols)
           n_elements = len(unique_elements)

           # Store properties
           props = {
               "n_atoms": n_atoms,
               "volume": volume,
               "density": density,
               "n_elements": n_elements,
               "formula": atoms.get_chemical_formula(),
           }
           properties.append(props)

       return properties


   # Add properties
   custom_props = add_custom_properties(structures)

   # Display properties
   for i, (atoms, props) in enumerate(zip(structures, custom_props)):
       print(f"Structure {i+1}: {props}")

****************************
 Advanced Analysis Examples
****************************

Chemical Space Exploration
==========================

.. code:: python

   def explore_chemical_space():
       """Explore different chemical compositions"""

       # Create diverse structures
       structures = []
       compositions = []

       # Binary compounds
       binary_pairs = [("Si", "C"), ("Si", "Ge"), ("C", "N"), ("B", "N")]

       for elem1, elem2 in binary_pairs:
           # Create simple binary structure (simplified)
           atoms1 = bulk(elem1, cubic=True, crystalstructure="diamond")
           atoms2 = bulk(elem2, cubic=True, crystalstructure="diamond")

           structures.extend([atoms1, atoms2])
           compositions.extend([elem1, elem2])

       # Extract features
       features = featurizer(structures, None)

       # Visualize chemical space
       plt.figure(figsize=(10, 8))

       # Color by element type
       element_colors = {
           "Si": "blue",
           "C": "black",
           "Ge": "green",
           "N": "red",
           "B": "orange",
       }

       colors = [element_colors.get(comp, "gray") for comp in compositions]

       scatter = plt.scatter(features[:, 0], features[:, 1], c=colors, s=100, alpha=0.7)

       # Add labels
       for i, comp in enumerate(compositions):
           plt.annotate(
               comp,
               (features[i, 0], features[i, 1]),
               xytext=(5, 5),
               textcoords="offset points",
           )

       plt.xlabel("Feature 1")
       plt.ylabel("Feature 2")
       plt.title("Chemical Space Exploration")
       plt.grid(True, alpha=0.3)
       plt.show()

       return structures, features, compositions


   chem_structures, chem_features, chem_compositions = explore_chemical_space()

Structural Motif Analysis
=========================

.. code:: python

   def analyze_structural_motifs():
       """Analyze different structural motifs"""

       structures = []
       motif_labels = []

       # Different crystal structures of silicon
       si_diamond = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
       si_fcc = bulk("Si", cubic=True, a=5.43, crystalstructure="fcc")
       si_bcc = bulk("Si", cubic=True, a=5.43, crystalstructure="bcc")

       structures.extend([si_diamond, si_fcc, si_bcc])
       motif_labels.extend(["diamond", "fcc", "bcc"])

       # Different coordination environments
       from ase.build import surface

       si_surface = surface("Si", (1, 0, 0), 4, vacuum=10.0)

       structures.append(si_surface)
       motif_labels.append("surface")

       # Molecular structures
       molecules = [molecule("H2O"), molecule("CO2"), molecule("CH4"), molecule("NH3")]
       mol_names = ["H2O", "CO2", "CH4", "NH3"]

       structures.extend(molecules)
       motif_labels.extend(mol_names)

       # Extract features
       features = featurizer(structures, None)

       # Visualize structural motifs
       plt.figure(figsize=(12, 8))

       # Create color map for motifs
       unique_motifs = list(set(motif_labels))
       colors = plt.cm.tab10(np.linspace(0, 1, len(unique_motifs)))
       motif_colors = {motif: colors[i] for i, motif in enumerate(unique_motifs)}

       for motif in unique_motifs:
           mask = [label == motif for label in motif_labels]
           motif_features = features[mask]

           plt.scatter(
               motif_features[:, 0],
               motif_features[:, 1],
               c=[motif_colors[motif]],
               s=100,
               alpha=0.7,
               label=motif,
           )

       plt.xlabel("Feature 1")
       plt.ylabel("Feature 2")
       plt.title("Structural Motif Analysis")
       plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
       plt.grid(True, alpha=0.3)
       plt.tight_layout()
       plt.show()

       return structures, features, motif_labels


   motif_structures, motif_features, motif_labels = analyze_structural_motifs()

Time Series Analysis
====================

.. code:: python

   def analyze_trajectory_evolution():
       """Analyze evolution of structures over time"""

       # Create a trajectory with gradual changes
       trajectory = []
       times = []

       # Simulate heating trajectory
       for i, temp in enumerate(np.linspace(0, 1000, 20)):
           # Create structure with thermal expansion (simplified)
           expansion = 1 + temp * 1e-5  # Simple thermal expansion
           atoms = bulk("Si", cubic=True, a=5.43 * expansion, crystalstructure="diamond")

           # Add some random displacement to simulate thermal motion
           positions = atoms.get_positions()
           displacement = np.random.normal(0, temp * 1e-4, positions.shape)
           atoms.set_positions(positions + displacement)

           trajectory.append(atoms)
           times.append(temp)

       # Extract features
       traj_features = featurizer(trajectory, None)

       # Visualize trajectory evolution
       plt.figure(figsize=(12, 5))

       plt.subplot(1, 2, 1)
       # Color by time/temperature
       scatter = plt.scatter(
           traj_features[:, 0],
           traj_features[:, 1],
           c=times,
           s=50,
           alpha=0.7,
           cmap="viridis",
       )
       plt.colorbar(scatter, label="Temperature (K)")
       plt.xlabel("Feature 1")
       plt.ylabel("Feature 2")
       plt.title("Trajectory in Feature Space")

       # Plot trajectory path
       plt.plot(traj_features[:, 0], traj_features[:, 1], "k-", alpha=0.3)

       plt.subplot(1, 2, 2)
       # Feature evolution over time
       plt.plot(times, traj_features[:, 0], "b-", label="Feature 1")
       plt.plot(times, traj_features[:, 1], "r-", label="Feature 2")
       plt.xlabel("Temperature (K)")
       plt.ylabel("Feature Value")
       plt.title("Feature Evolution")
       plt.legend()
       plt.grid(True, alpha=0.3)

       plt.tight_layout()
       plt.show()

       return trajectory, traj_features, times


   traj_structures, traj_features, traj_times = analyze_trajectory_evolution()

*********************
 Similarity Analysis
*********************

Structure Similarity Metrics
============================

.. code:: python

   from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


   def analyze_structure_similarity(structures, features):
       """Analyze similarity between structures"""

       # Calculate similarity matrices
       cosine_sim = cosine_similarity(features)
       euclidean_dist = euclidean_distances(features)

       # Create labels
       labels = [atoms.get_chemical_formula() for atoms in structures]

       # Plot similarity matrices
       fig, axes = plt.subplots(1, 2, figsize=(12, 5))

       # Cosine similarity
       im1 = axes[0].imshow(cosine_sim, cmap="viridis")
       axes[0].set_title("Cosine Similarity")
       axes[0].set_xticks(range(len(labels)))
       axes[0].set_yticks(range(len(labels)))
       axes[0].set_xticklabels(labels, rotation=45)
       axes[0].set_yticklabels(labels)
       plt.colorbar(im1, ax=axes[0])

       # Euclidean distance
       im2 = axes[1].imshow(euclidean_dist, cmap="viridis_r")
       axes[1].set_title("Euclidean Distance")
       axes[1].set_xticks(range(len(labels)))
       axes[1].set_yticks(range(len(labels)))
       axes[1].set_xticklabels(labels, rotation=45)
       axes[1].set_yticklabels(labels)
       plt.colorbar(im2, ax=axes[1])

       plt.tight_layout()
       plt.show()

       return cosine_sim, euclidean_dist


   # Analyze similarity for our structures
   cos_sim, euc_dist = analyze_structure_similarity(structures, features)

Nearest Neighbors Analysis
==========================

.. code:: python

   from sklearn.neighbors import NearestNeighbors


   def find_similar_structures(structures, features, query_idx, n_neighbors=3):
       """Find most similar structures to a query structure"""

       # Fit nearest neighbors
       nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean")
       nn.fit(features)

       # Find neighbors for query structure
       distances, indices = nn.kneighbors([features[query_idx]])

       # Remove self (first neighbor)
       distances = distances[0][1:]
       indices = indices[0][1:]

       # Print results
       query_formula = structures[query_idx].get_chemical_formula()
       print(f"Most similar structures to {query_formula}:")

       for i, (dist, idx) in enumerate(zip(distances, indices)):
           similar_formula = structures[idx].get_chemical_formula()
           print(f"  {i+1}. {similar_formula} (distance: {dist:.3f})")

       return indices, distances


   # Find structures similar to water
   if len(structures) > 3:  # Make sure we have enough structures
       water_idx = 3  # Assuming water is at index 3
       similar_indices, similar_distances = find_similar_structures(
           structures, features, water_idx, n_neighbors=2
       )

****************************
 Export and Data Management
****************************

Saving Features and Results
===========================

.. code:: python

   import pickle
   import json


   def save_exploration_results(structures, features, filename_base):
       """Save exploration results for later use"""

       # Save features
       np.save(f"{filename_base}_features.npy", features)

       # Save structure information
       structure_info = []
       for i, atoms in enumerate(structures):
           info = {
               "index": i,
               "formula": atoms.get_chemical_formula(),
               "n_atoms": len(atoms),
               "cell": atoms.get_cell().tolist(),
               "positions": atoms.get_positions().tolist(),
               "symbols": atoms.get_chemical_symbols(),
           }
           structure_info.append(info)

       with open(f"{filename_base}_structures.json", "w") as f:
           json.dump(structure_info, f, indent=2)

       # Save complete structures (for reconstruction)
       with open(f"{filename_base}_atoms.pkl", "wb") as f:
           pickle.dump(structures, f)

       print(f"Saved exploration results to {filename_base}_*")


   def load_exploration_results(filename_base):
       """Load previously saved exploration results"""

       # Load features
       features = np.load(f"{filename_base}_features.npy")

       # Load structure information
       with open(f"{filename_base}_structures.json", "r") as f:
           structure_info = json.load(f)

       # Load complete structures
       with open(f"{filename_base}_atoms.pkl", "rb") as f:
           structures = pickle.load(f)

       return structures, features, structure_info


   # Example usage
   # save_exploration_results(structures, features, "my_exploration")
   # loaded_structures, loaded_features, loaded_info = load_exploration_results("my_exploration")

****************
 Best Practices
****************

Choosing Appropriate Datasets
=============================

#. **Diversity**: Include diverse chemical compositions and structures
#. **Size**: Balance dataset size with computational resources
#. **Relevance**: Focus on chemically meaningful comparisons
#. **Quality**: Ensure structures are reasonable and well-optimized

Interpretation Guidelines
=========================

#. **Feature space**: Remember that features are learned
   representations, not physical properties
#. **Clustering**: Clusters may reflect chemical similarity, structural
   similarity, or both
#. **Outliers**: Unusual positions in feature space may indicate novel
   structures or errors
#. **Validation**: Cross-reference feature-based similarities with known
   chemical knowledge

Performance Optimization
========================

.. code:: python

   # For large datasets, consider:

   # 1. Batch processing
   featurizer = PETMADFeaturizer(
       version="latest",
       device="cuda",  # Use GPU if available
       batch_size=32,  # Adjust based on memory
   )

   # 2. Progress tracking
   from tqdm import tqdm

   featurizer_with_progress = PETMADFeaturizer(
       version="latest", progress_bar=tqdm  # Show progress for large datasets
   )

*****************
 Troubleshooting
*****************

Common Issues
=============

#. **Memory errors**: Reduce batch size or use CPU instead of GPU
#. **Inconsistent results**: Ensure all structures use the same
   coordinate system
#. **Poor clustering**: Try different distance metrics or preprocessing

Validation Checks
=================

.. code:: python

   def validate_exploration_results(structures, features):
       """Validate exploration results"""

       # Check for NaN or infinite values
       if np.any(np.isnan(features)) or np.any(np.isinf(features)):
           print("Warning: Features contain NaN or infinite values")

       # Check feature variance
       feature_std = np.std(features, axis=0)
       if np.any(feature_std < 1e-6):
           print("Warning: Some features have very low variance")

       # Check structure consistency
       for i, atoms in enumerate(structures):
           if len(atoms) == 0:
               print(f"Warning: Structure {i} is empty")

       print("Validation completed")


   # Validate results
   validate_exploration_results(structures, features)
