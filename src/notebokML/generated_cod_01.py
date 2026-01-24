import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

def generate_synthetic_data(d=5, n=1000):
    """
    Generuje dane syntetyczne dla prostego DAGa.
    d: liczba węzłów (zmiennych)
    n: liczba próbek
    Na podstawie metodologii opisanej w sekcji eksperymentów [3].
    """
    # 1. Tworzenie macierzy sąsiedztwa (prawdziwy graf - ground truth)
    # Tworzymy macierz górnotrójkątną, aby zagwarantować acykliczność (DAG)
    W_true = np.triu(np.random.uniform(0.5, 2.0, (d, d)), k=1)
    
    # Losowe zerowanie niektórych krawędzi dla rzadkości grafu
    W_true[np.random.rand(d, d) < 0.5] = 0
    
    # 2. Generowanie danych X = XW + Z (Liniowy SEM)
    # Zgodnie z równaniem X_j = w_j^T * X + z_j [4]
    noise = np.random.normal(0, 1, (n, d))
    I = np.eye(d)
    # Rozwiązanie równania X(I - W) = Z => X = Z(I - W)^-1
    X = noise @ np.linalg.inv(I - W_true)
    
    return X, W_true

def notears_algorithm(X, rho_max=1e16, w_threshold=0.3):
    """
    Implementacja algorytmu NOTEARS.
    Źródło: Algorithm 1 w 'DAGs with NO TEARS' [5].
    """
    n, d = X.shape
    W_est = np.zeros((d, d)) # Inicjalizacja W0
    rho = 1.0                # Parametr kary (penalty parameter)
    alpha = 0.0              # Mnożnik Lagrange'a
    h_val = np.inf           # Wartość ograniczenia acykliczności

    # Kluczowa funkcja h(W): miara "acykliczności" grafu.
    # h(W) = tr(e^(W*W)) - d = 0 oznacza, że graf jest DAGiem [6].
    def _h(W):
        return np.trace(expm(W * W)) - d

    # Funkcja straty F(W): Błąd średniokwadratowy (Least Squares)
    # F(W) = 1/2n || X - XW ||^2 [7]
    def _loss(W_flat):
        W = W_flat.reshape(d, d)
        R = X - X @ W
        loss = 0.5 / n * np.sum(R ** 2)
        return loss

    # Funkcja celu dla rozszerzonego Lagrange'a (Augmented Lagrangian) [8]
    # L(W, alpha, rho) = F(W) + rho/2 * |h(W)|^2 + alpha * h(W)
    def _func(W_flat):
        W = W_flat.reshape(d, d)
        h = _h(W)
        return _loss(W_flat) + 0.5 * rho * h * h + alpha * h

    # Gradient funkcji celu (uproszczony - numeryczny dla czytelności,
    # w pełnej wersji implementuje się analityczny gradient [6])
    
    # Pętla główna algorytmu (Dual Ascent) [5]
    while rho < rho_max:
        # Krok 2(a): Rozwiązanie problemu pierwotnego
        res = minimize(_func, W_est.flatten(), method='L-BFGS-B')
        W_new = res.x.reshape(d, d)
        h_new = _h(W_new)
        
        # Sprawdzenie warunku zbieżności
        if h_new > 0.25 * h_val:
            rho *= 10  # Zwiększenie kary
        else:
            W_est = W_new
            h_val = h_new
            alpha += rho * h_val # Krok 2(b): Dual ascent [5]
            
        if h_val < 1e-8:
            break

    # Krok 3: Progowanie (Thresholding) [9]
    # Usuwamy wagi bliskie zera, aby uzyskać czystą strukturę
    W_est[np.abs(W_est) < w_threshold] = 0
    
    return W_est


def visualize_results(W_true, W_est):
    """
    Wizualizuje porównanie prawdziwej i nauczonej struktury grafu.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Heatmapa prawdziwa
    sns.heatmap(W_true, annot=True, fmt=".2f", cmap="Blues", ax=axes[0, 0])
    axes[0, 0].set_title("Prawdziwa macierz wag (W_true)")
    
    # 2. Heatmapa nauczona
    sns.heatmap(W_est, annot=True, fmt=".2f", cmap="Greens", ax=axes[0, 1])
    axes[0, 1].set_title("Nauczona macierz wag (W_est)")
    
    # Próg wyswietlania krawędzi dla grafów
    threshold = 0.1
    
    # 3. Graf prawdziwy
    G_true = nx.DiGraph(W_true)
    # Usuwamy słabe krawędzie z wizualizacji (choć w W_true raczej są 0 lub >0.5)
    edges_true = [(u, v) for u, v, d in G_true.edges(data=True) if abs(d['weight']) > 0]
    pos = nx.circular_layout(G_true)
    
    nx.draw_networkx_nodes(G_true, pos, ax=axes[1, 0], node_color='lightblue', node_size=500)
    nx.draw_networkx_labels(G_true, pos, ax=axes[1, 0])
    nx.draw_networkx_edges(G_true, pos, edgelist=edges_true, ax=axes[1, 0], edge_color='blue', arrows=True)
    axes[1, 0].set_title("Prawdziwy graf")
    axes[1, 0].axis('off')

    # 4. Graf nauczony
    G_est = nx.DiGraph(W_est)
    edges_est = [(u, v) for u, v, d in G_est.edges(data=True) if abs(d['weight']) > threshold]
    
    nx.draw_networkx_nodes(G_est, pos, ax=axes[1, 1], node_color='lightgreen', node_size=500)
    nx.draw_networkx_labels(G_est, pos, ax=axes[1, 1])
    nx.draw_networkx_edges(G_est, pos, edgelist=edges_est, ax=axes[1, 1], edge_color='green', arrows=True)
    axes[1, 1].set_title("Nauczony graf")
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

# --- Uruchomienie przykładu ---
if __name__ == "__main__":
    # 1. Generowanie danych
    print("Generowanie danych syntetycznych...")
    X, W_true = generate_synthetic_data(d=5, n=1000)
    
    # 2. Uczenie struktury
    print("Uczenie struktury DAG (NOTEARS)...")
    W_learned = notears_algorithm(X)
    
    # 3. Wyniki
    print("\nPrawdziwa macierz wag (W_true):")
    print(np.round(W_true, 2))
    
    print("\nNauczona macierz wag (W_learned):")
    print(np.round(W_learned, 2))
    
    # Sprawdzenie błędu
    # Proste porównanie (w praktyce używa się miary SHD - Structural Hamming Distance)
    mse = np.sum((W_true - W_learned)**2)
    print(f"\nBłąd rekonstrukcji macierzy (MSE): {mse:.4f}")

    # 4. Wizualizacja
    print("Generowanie wykresów...")
    visualize_results(W_true, W_learned)