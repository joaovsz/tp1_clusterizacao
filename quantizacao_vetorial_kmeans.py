import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import pandas as pd

def main():
    print("=== QUANTIZAÇÃO VETORIAL COM K-MEANS ===\n")
    
    # 1. Gerar dataset sintético make_blobs
    print("📊 Gerando dataset sintético com make_blobs...")
    X, y_true = make_blobs(
        n_samples=300,
        centers=4,
        n_features=2,
        random_state=42,
        cluster_std=1.5
    )
    
    print(f"Dataset gerado: {X.shape[0]} pontos, {X.shape[1]} dimensões")
    print(f"Centros reais: {len(np.unique(y_true))} clusters")
    
    # 2. Aplicar K-means para quantização
    print("\n🎯 Aplicando K-means para quantização vetorial...")
    
    # Testar diferentes números de clusters (níveis de quantização)
    k_values = [2, 4, 8, 16]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    results = {}
    
    for idx, k in enumerate(k_values):
        print(f"\n📐 Quantizando com K={k} clusters:")
        
        # Aplicar K-means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        centroids = kmeans.cluster_centers_
        
        # Quantizar os dados (substituir cada ponto pelo centróide mais próximo)
        X_quantized = centroids[labels]
        
        # Calcular erro de quantização (distorção)
        distortion = np.sum((X - X_quantized) ** 2) / len(X)
        
        results[k] = {
            'labels': labels,
            'centroids': centroids,
            'X_quantized': X_quantized,
            'distortion': distortion,
            'inertia': kmeans.inertia_
        }
        
        print(f"  Distorção média: {distortion:.4f}")
        print(f"  Inércia: {kmeans.inertia_:.4f}")
        print(f"  Redução de dados: {len(X)} pontos → {k} representantes")
        
        # Plotar resultados
        ax = axes[idx]
        
        # Pontos originais (cinza claro)
        ax.scatter(X[:, 0], X[:, 1], c='lightgray', alpha=0.6, s=30, label='Dados originais')
        
        # Pontos quantizados (coloridos)
        colors = plt.cm.tab10(np.linspace(0, 1, k))
        for i in range(k):
            mask = labels == i
            ax.scatter(X_quantized[mask, 0], X_quantized[mask, 1], 
                      c=[colors[i]], s=50, alpha=0.8, label=f'Cluster {i}')
        
        # Centróides (estrelas pretas)
        ax.scatter(centroids[:, 0], centroids[:, 1], 
                  c='black', marker='*', s=200, label='Centróides')
        
        # Linhas conectando pontos originais aos quantizados
        for i in range(0, len(X), 10):  # Mostrar algumas conexões
            ax.plot([X[i, 0], X_quantized[i, 0]], [X[i, 1], X_quantized[i, 1]], 
                   'k--', alpha=0.3, linewidth=0.5)
        
        ax.set_title(f'K-means: K={k} clusters\nDistorção: {distortion:.4f}')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    plt.suptitle('Quantização Vetorial com K-means - Dataset make_blobs', 
                 fontsize=16, y=1.02)
    plt.savefig('quantizacao_vetorial_clusters.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Análise comparativa
    print("\n📊 ANÁLISE COMPARATIVA DOS RESULTADOS:")
    print(f"{'K':<5} {'Distorção':<12} {'Inércia':<12} {'Taxa Compressão':<15} {'Bits/Ponto':<12}")
    print("-" * 70)
    
    for k in k_values:
        result = results[k]
        compression_ratio = len(X) / k
        bits_per_point = np.log2(k)
        
        print(f"{k:<5} {result['distortion']:<12.4f} {result['inertia']:<12.2f} "
              f"{compression_ratio:<15.2f} {bits_per_point:<12.2f}")
    
    # 4. Curva de distorção vs K
    print("\n📈 Criando curva de distorção...")
    
    plt.figure(figsize=(10, 6))
    
    # Plot da distorção
    k_range = range(1, 21)
    distortions = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        centroids = kmeans.cluster_centers_
        X_quantized = centroids[labels]
        distortion = np.sum((X - X_quantized) ** 2) / len(X)
        distortions.append(distortion)
    
    plt.subplot(1, 2, 1)
    plt.plot(k_range, distortions, 'bo-')
    plt.xlabel('Número de Clusters (K)')
    plt.ylabel('Distorção Média')
    plt.title('Curva de Distorção - Quantização Vetorial')
    plt.grid(True, alpha=0.3)
    
    # Plot dos bits por ponto
    plt.subplot(1, 2, 2)
    bits_per_point = [np.log2(k) for k in k_range]
    plt.plot(k_range, bits_per_point, 'ro-')
    plt.xlabel('Número de Clusters (K)')
    plt.ylabel('Bits por Ponto')
    plt.title('Taxa de Bits vs K')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('curva_distorcao_quantizacao.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Exemplo prático de quantização
    print("\n💡 EXEMPLO PRÁTICO DE QUANTIZAÇÃO:")
    
    # Pegar alguns pontos originais e mostrar sua quantização
    sample_indices = [0, 50, 100, 150, 200]
    k_demo = 4
    result_demo = results[k_demo]
    
    print(f"\nQuantização com K={k_demo} clusters:")
    print("Ponto Original → Ponto Quantizado (Cluster)")
    print("-" * 50)
    
    for idx in sample_indices:
        original = X[idx]
        quantized = result_demo['X_quantized'][idx]
        cluster = result_demo['labels'][idx]
        error = np.linalg.norm(original - quantized)
        
        print(f"({original[0]:.2f}, {original[1]:.2f}) → "
              f"({quantized[0]:.2f}, {quantized[1]:.2f}) [C{cluster}] "
              f"Erro: {error:.3f}")
    
    # 6. Salvar resultados
    print(f"\n💾 Salvando resultados...")
    
    # Criar DataFrame com resultados da melhor quantização
    best_k = 4  # Escolha baseada no trade-off
    best_result = results[best_k]
    
    df_results = pd.DataFrame({
        'x_original': X[:, 0],
        'y_original': X[:, 1],
        'x_quantized': best_result['X_quantized'][:, 0],
        'y_quantized': best_result['X_quantized'][:, 1],
        'cluster': best_result['labels'],
        'erro_quantizacao': [np.linalg.norm(X[i] - best_result['X_quantized'][i]) 
                            for i in range(len(X))]
    })
    
    df_results.to_csv('quantizacao_vetorial_resultados.csv', index=False)
    print(f"✅ Resultados salvos em: quantizacao_vetorial_resultados.csv")
    
    # 7. Estatísticas finais
    print(f"\n📋 ESTATÍSTICAS FINAIS (K={best_k}):")
    print(f"Número de pontos originais: {len(X)}")
    print(f"Número de representantes: {best_k}")
    print(f"Taxa de compressão: {len(X)/best_k:.1f}:1")
    print(f"Bits por ponto: {np.log2(best_k):.1f} bits")
    print(f"Distorção média: {best_result['distortion']:.4f}")
    print(f"Erro máximo: {df_results['erro_quantizacao'].max():.3f}")
    print(f"Erro médio: {df_results['erro_quantizacao'].mean():.3f}")
    
    print(f"\n🎯 Quantização vetorial concluída com sucesso!")

if __name__ == "__main__":
    main()
