import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import pairwise_distances_argmin_min, silhouette_score, calinski_harabasz_score, davies_bouldin_score, accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


your_dataset_path = 'iris.data'
your_dataset = pd.read_csv(your_dataset_path)
your_numeric_dataset = your_dataset.select_dtypes(include=[np.number])


X = your_numeric_dataset.iloc[:, :-1].values
class_labels = your_dataset.iloc[:, -1].values


label_encoder = LabelEncoder()
actual_labels = label_encoder.fit_transform(class_labels)


num_clusters = 4
population_size = 800
num_generations = 50
mutation_rate = 0.2

scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)


def initialize_population(population_size, num_clusters, data):
    return np.random.rand(population_size, num_clusters, data.shape[1])


def fitness(population, data):
    distances = np.array([pairwise_distances_argmin_min(data, centroids)[1] for centroids in population])
    return -np.sum(distances, axis=1)

def crossover(parent1, parent2):
    mask = np.random.rand(*parent1.shape) < 0.5
    child1 = np.where(mask, parent1, parent2)
    child2 = np.where(mask, parent2, parent1)
    return child1, child2


def mutate(child, mutation_rate):
    mask = np.random.rand(*child.shape) < mutation_rate
    child += mask * np.random.randn(*child.shape)
    return child


def genetic_algorithm(data, num_clusters, population_size, num_generations, mutation_rate):
    population = initialize_population(population_size, num_clusters, data)
    fitness_history = []
    for generation in range(num_generations):

        fitness_values = fitness(population, data)
        fitness_history.append(np.max(fitness_values))

        selected_indices = np.argsort(fitness_values)[-population_size // 2:]
        selected_population = population[selected_indices]

        new_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = selected_population[
                np.random.choice(selected_population.shape[0], size=2, replace=False)]
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([child1, child2])

        new_population = np.array(new_population)[:population_size // 2]

        new_population = np.array([mutate(child, mutation_rate) for child in new_population])

        population[population_size // 2:] = new_population

        best_fitness = np.max(fitness_values)
        print(f"Generation {generation + 1}, Best Fitness: {best_fitness}")

    best_solution = population[np.argmax(fitness_values)]
    return best_solution, fitness_history

best_solution, fitness_history = genetic_algorithm(X_normalized, num_clusters, population_size, num_generations, mutation_rate)

cluster_assignments = pairwise_distances_argmin_min(X_normalized, best_solution)[0]

print("Final Cluster Assignments:", cluster_assignments)

silhouette_avg = silhouette_score(X_normalized, cluster_assignments)
ch_score = calinski_harabasz_score(X_normalized, cluster_assignments)
db_score = davies_bouldin_score(X_normalized, cluster_assignments)
print(f"Silhouette Score: {silhouette_avg}")
print(f"Calinski-Harabasz Index: {ch_score}")
print(f"Davies-Bouldin Index: {db_score}")
plt.plot(fitness_history)
plt.title('Fitness Scores Over Generations')
plt.xlabel('Generation')
plt.ylabel('Fitness Score')
plt.show()

accuracy = accuracy_score(actual_labels, cluster_assignments)
print("Correctly Predicted Ratio:", accuracy)

comparison_df = pd.DataFrame({'Actual': actual_labels, 'Predicted': cluster_assignments})

print("Comparison DataFrame:\n",comparison_df.head())

conf_matrix = confusion_matrix(actual_labels, cluster_assignments)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels='auto', yticklabels='auto')
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.title('Confusion Matrix')
plt.show()

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_normalized)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=actual_labels, cmap='viridis', edgecolors='k', s=50)
plt.title('Clustering Results with GA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_assignments)
plt.show()

kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init='auto').fit(X_pca)
kmeans_labels = kmeans.labels_

if actual_labels is not None:
    accuracy_kmeans = accuracy_score(actual_labels, kmeans_labels)
    print("Accuracy Score of KMeans:", accuracy_kmeans)


plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels)
plt.title('Clustering Results with KMeans')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()