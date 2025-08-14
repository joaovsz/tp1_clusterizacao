import kagglehub
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

def main():
    print("=== QUESTÃO 5: PCA + K-MEANS NO DATASET DA SHOPMANIA ===\n")
    
    print("Baixando dataset da ShopMania...")
    path = kagglehub.dataset_download("lakritidis/product-classification-and-categorization")
    
    files = os.listdir(path)
    csv_files = [f for f in files if f.endswith('.csv')]
    
    # Usar shopmania.csv se existir (melhor estrutura)
    target_file = None
    for f in csv_files:
        if 'shopmania' in f.lower():
            target_file = f
            break
    if not target_file:
        target_file = csv_files[0]
    
    df = pd.read_csv(os.path.join(path, target_file))
    print(f"Dataset original: {df.shape[0]} linhas, {df.shape[1]} colunas")
    
    sample_size = min(15000, len(df))  
    if len(df) > sample_size:
        print(f"⚡ Usando amostra de {sample_size} produtos para análise RÁPIDA...")
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    
    print(f"Dataset para análise: {df.shape[0]} linhas")
    
    # 3. Preparar dados
    print("\n=== PREPARANDO DADOS ===")
    text_columns = [col for col in df.columns if df[col].dtype == 'object' and col != 'category']
    
    if text_columns:
        text_col = text_columns[0]
        text_data = df[text_col].fillna('').astype(str)
        
        # OTIMIZAÇÃO: TF-IDF com menos features e mantendo matriz esparsa
        print("🔄 Aplicando TF-IDF otimizado...")
        vectorizer = TfidfVectorizer(
            max_features=500,  
            stop_words='english',
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 1) 
        )
        X_sparse = vectorizer.fit_transform(text_data)  # Mantém matriz esparsa
        print(f"✅ Matriz TF-IDF esparsa criada: {X_sparse.shape}")
        
        # Converter para densa só quando necessário e em partes menores
        X_original = X_sparse.toarray()
        print(f"Matriz densa: {X_original.shape}")
    else:
        print("❌ Nenhuma coluna de texto encontrada!")
        return
    
    # 4. Aplicar PCA (OTIMIZADO)
    print("\n=== APLICANDO PCA ===")
    
    n_components_list = [20, 50, 100] 
    results = {}
    
    print("⚡ Executando PCA otimizado...")
    for n_comp in n_components_list:
        if n_comp >= X_original.shape[1]:
            print(f"❌ Pulando {n_comp} (maior que {X_original.shape[1]} features)")
            continue
            
        print(f"\n🔄 PCA com {n_comp} componentes:")
        
        # Aplicar PCA
        pca = PCA(n_components=n_comp, random_state=42) 
        X_pca = pca.fit_transform(X_original)
        
        # Variância explicada
        explained_variance = pca.explained_variance_ratio_.sum()
        print(f"Variância explicada: {explained_variance:.3f}")
        print(f"Redução: {X_original.shape[1]} → {X_pca.shape[1]} dimensões")
        
        k = 5
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
        labels = kmeans.fit_predict(X_pca)
        
        results[n_comp] = {
            'X_pca': X_pca,
            'labels': labels,
            'inertia': kmeans.inertia_,
            'explained_variance': explained_variance,
            'pca': pca
        }

        print(f"K-means inércia: {kmeans.inertia_:.2f}")

        # Distribuição rápida
        unique, counts = np.unique(labels, return_counts=True)
        cluster_info = [f"C{i}:{c}" for i, c in zip(unique, counts)]
        print(f"Distribuição: {' | '.join(cluster_info)}")

    # 5. K-means nos dados originais 
    print(f"\n=== K-MEANS ORIGINAL (sem PCA) ===")
    print("🔄 Executando K-means nos dados originais...")
    kmeans_original = KMeans(n_clusters=5, random_state=42, n_init=5)
    labels_original = kmeans_original.fit_predict(X_original)
    print(f"Inércia sem PCA: {kmeans_original.inertia_:.2f}")
    
    # 6. Comparação RÁPIDA dos resultados
    print(f"\n=== 📊 COMPARAÇÃO DOS RESULTADOS ===")
    print(f"{'Método':<20} {'Dimensões':<10} {'Variância':<10} {'Inércia':<12} {'Tempo':<8}")
    print("-" * 65)
    
    print(f"{'Original':<20} {X_original.shape[1]:<10} {'1.000':<10} {kmeans_original.inertia_:<12.2f} {'Base':<8}")
    
    for n_comp in sorted(results.keys()):
        result = results[n_comp]
        reduction = (1 - n_comp/X_original.shape[1]) * 100
        print(f"{f'PCA-{n_comp}':<20} {n_comp:<10} {result['explained_variance']:<10.3f} {result['inertia']:<12.2f} {f'-{reduction:.0f}%':<8}")
    
    print(f"\n=== 📈 VISUALIZAÇÃO 2D ===")
    print("🔄 Criando visualização PCA 2D...")
    
    pca_2d = PCA(n_components=2, random_state=42)
    X_2d = pca_2d.fit_transform(X_original)
    
    # K-means em 2D
    kmeans_2d = KMeans(n_clusters=5, random_state=42, n_init=5)
    labels_2d = kmeans_2d.fit_predict(X_2d)
    
    # Plot otimizado
    try:
        plt.figure(figsize=(10, 6))
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        
        plot_sample_size = min(5000, len(X_2d))
        if len(X_2d) > plot_sample_size:
            idx = np.random.choice(len(X_2d), plot_sample_size, replace=False)
            X_2d_plot = X_2d[idx]
            labels_2d_plot = labels_2d[idx]
        else:
            X_2d_plot = X_2d
            labels_2d_plot = labels_2d
        
        for i in range(5):
            mask = labels_2d_plot == i
            plt.scatter(X_2d_plot[mask, 0], X_2d_plot[mask, 1], 
                       c=colors[i], label=f'Cluster {i}', alpha=0.6, s=20)
        
        var1, var2 = pca_2d.explained_variance_ratio_
        plt.xlabel(f'PC1 ({var1:.3f} var)')
        plt.ylabel(f'PC2 ({var2:.3f} var)')
        plt.title('K-means com PCA (2D) - Amostra')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        print("Visualização criada!")
    except Exception as e:
        print(f"Plot não disponível: {e}")
    
    best_n_comp = min(results.keys()) if results else None
    
    if best_n_comp:
        print(f"\n===ANÁLISE FINAL ===")
        best_result = results[best_n_comp]
        
        # Adicionar colunas de cluster
        df['cluster_pca'] = best_result['labels']
        df['cluster_original'] = labels_original
        
        output_file = 'resultados_pca_kmeans.csv'
        df.to_csv(output_file, index=False)
        print(f"💾 Resultados salvos: {output_file}")
        
        # Estatísticas finais
        print(f"🏆 Melhor configuração: PCA com {best_n_comp} componentes")
        print(f"📏 Redução dimensional: {X_original.shape[1]} → {best_n_comp} ({(1-best_n_comp/X_original.shape[1])*100:.0f}% redução)")
        print(f"📊 Variância preservada: {best_result['explained_variance']:.3f}")
        
        inertia_diff = kmeans_original.inertia_ - best_result['inertia']
        if inertia_diff > 0:
            print(f"📈 Melhoria inércia: -{inertia_diff:.2f} (PCA melhorou!)")
        else:
            print(f"📉 Mudança inércia: {inertia_diff:+.2f} (trade-off dimensional)")
        
        print(f"⚡ Análise concluída em dataset de {len(df)} produtos!")
    
    else:
        print("❌ Nenhum resultado PCA válido encontrado")

if __name__ == "__main__":
    main()
