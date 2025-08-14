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
    
    # Usar mesmo tamanho de amostra da Questão 2 para comparação válida
    if len(df) > 50000:
        print(f"Dataset grande ({len(df)} linhas). Usando amostra de 50,000 para comparação com Questão 2...")
        sample_idx = np.random.choice(len(df), 50000, replace=False)
        df = df.iloc[sample_idx].reset_index(drop=True)
        print(f"Nova amostra: {len(df)} produtos")
    
    print(f"Dataset para análise: {df.shape[0]} linhas (mesmo critério da Questão 2)")
    
    # 3. Preparar dados (usando mesmos parâmetros da Questão 2)
    print("\n=== PREPARANDO DADOS ===")
    text_columns = [col for col in df.columns if df[col].dtype == 'object' and col != 'category']
    
    # Usar mesma lógica da Questão 2 para escolher coluna
    product_col = None
    possible_names = ['product', 'name', 'title', 'description']
    
    for col in text_columns:
        col_lower = col.lower()
        if any(name in col_lower for name in possible_names):
            product_col = col
            break
    
    # Se não encontrou, usar coluna com mais diversidade
    if not product_col and text_columns:
        text_diversity = {}
        for col in text_columns[:3]:
            if col.lower() != 'category':
                unique_ratio = df[col].nunique() / len(df)
                text_diversity[col] = unique_ratio
        
        if text_diversity:
            product_col = max(text_diversity, key=text_diversity.get)
            print(f"Coluna escolhida (maior diversidade): '{product_col}'")
    
    if text_columns:
        text_col = product_col if product_col else text_columns[0]
        text_data = df[text_col].fillna('').astype(str)
        
        # USAR MESMOS PARÂMETROS TF-IDF DA QUESTÃO 2
        print("🔄 Aplicando TF-IDF (mesmos parâmetros da Questão 2)...")
        vectorizer = TfidfVectorizer(
            max_features=1000,  # MESMO valor da Questão 2
            stop_words='english',
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 2)  # MESMO valor da Questão 2 (uni e bigramas)
        )
        X_sparse = vectorizer.fit_transform(text_data)
        
        # Converter para densa (mesmo que Questão 2)
        X_original = X_sparse.toarray()
        print(f"✅ Matriz TF-IDF criada: {X_original.shape} (mesma estrutura da Questão 2)")
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
        
        # Usar mesmos valores de K da Questão 2
        k_values_q2 = [3, 5, 8, 10]  # MESMOS valores da Questão 2
        
        print(f"Testando PCA com {n_comp} componentes nos mesmos valores de K da Questão 2: {k_values_q2}")
        
        # Aplicar K-means com cada valor de K da Questão 2
        k_results = {}
        for k in k_values_q2:
            kmeans = KMeans(n_clusters=k, random_state=42)  # MESMO parâmetro da Questão 2
            labels = kmeans.fit_predict(X_pca)
            k_results[k] = {
                'labels': labels,
                'inertia': kmeans.inertia_,
                'model': kmeans
            }
            print(f"  K={k}: Inércia={kmeans.inertia_:.2f}")
        
        # Encontrar melhor K usando mesmo critério da Questão 2 (método do cotovelo)
        inertias = [k_results[k]['inertia'] for k in k_values_q2]
        if len(inertias) > 2:
            drops = []
            for i in range(1, len(inertias)):
                drop = (inertias[i-1] - inertias[i]) / inertias[i-1]
                drops.append(drop)
            best_k_idx = drops.index(max(drops))
            best_k = k_values_q2[best_k_idx + 1]
        else:
            best_k = k_values_q2[1] if len(k_values_q2) > 1 else k_values_q2[0]
        
        print(f"  Melhor K (método do cotovelo): {best_k}")
        
        results[n_comp] = {
            'X_pca': X_pca,
            'k_results': k_results,
            'best_k': best_k,
            'best_labels': k_results[best_k]['labels'],
            'best_inertia': k_results[best_k]['inertia'],
            'explained_variance': explained_variance,
            'pca': pca
        }

    # 5. K-means nos dados originais (REPLICANDO QUESTÃO 2)
    print(f"\n=== K-MEANS ORIGINAL - REPLICANDO QUESTÃO 2 ===")
    k_values_q2 = [3, 5, 8, 10]
    original_results = {}
    
    print("🔄 Executando K-means nos dados originais com mesmos parâmetros da Questão 2...")
    for k in k_values_q2:
        print(f"\nTestando k={k}")
        kmeans_original = KMeans(n_clusters=k, random_state=42)  # MESMO da Questão 2
        labels_original = kmeans_original.fit_predict(X_original)
        
        original_results[k] = {
            'labels': labels_original,
            'inertia': kmeans_original.inertia_,
            'model': kmeans_original
        }
        
        print(f"Inércia sem PCA: {kmeans_original.inertia_:.2f}")
        
        # Distribuição dos clusters
        unique, counts = np.unique(labels_original, return_counts=True)
        for cluster_id, count in zip(unique, counts):
            print(f"  Cluster {cluster_id}: {count} produtos")
    
    # Encontrar melhor K para dados originais (método do cotovelo)
    inertias_original = [original_results[k]['inertia'] for k in k_values_q2]
    if len(inertias_original) > 2:
        drops = []
        for i in range(1, len(inertias_original)):
            drop = (inertias_original[i-1] - inertias_original[i]) / inertias_original[i-1]
            drops.append(drop)
        best_k_original_idx = drops.index(max(drops))
        best_k_original = k_values_q2[best_k_original_idx + 1]
    else:
        best_k_original = k_values_q2[1] if len(k_values_q2) > 1 else k_values_q2[0]
    
    print(f"\nMelhor K para dados originais (método do cotovelo): {best_k_original}")
    best_labels_original = original_results[best_k_original]['labels']
    best_inertia_original = original_results[best_k_original]['inertia']
    
    # 6. Comparação DETALHADA dos resultados (Questão 2 vs PCA)
    print(f"\n=== 📊 COMPARAÇÃO QUESTÃO 2 vs PCA ===")
    print(f"{'Método':<25} {'K Ótimo':<8} {'Dimensões':<10} {'Variância':<10} {'Inércia':<12} {'Melhoria':<10}")
    print("-" * 85)
    
    print(f"{'Original (Questão 2)':<25} {best_k_original:<8} {X_original.shape[1]:<10} {'1.000':<10} {best_inertia_original:<12.2f} {'Base':<10}")
    
    for n_comp in sorted(results.keys()):
        result = results[n_comp]
        best_k = result['best_k']
        best_inertia = result['best_inertia']
        
        # Calcular melhoria/deterioração
        inertia_change = ((best_inertia_original - best_inertia) / best_inertia_original) * 100
        change_str = f"{inertia_change:+.1f}%"
        
        print(f"{f'PCA-{n_comp}':<25} {best_k:<8} {n_comp:<10} {result['explained_variance']:<10.3f} {best_inertia:<12.2f} {change_str:<10}")
    
    # Adicionar análise do método do cotovelo
    print(f"\n=== 📈 MÉTODO DO COTOVELO - COMPARAÇÃO ===")
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Dados originais
    plt.subplot(1, 3, 1)
    plt.plot(k_values_q2, [original_results[k]['inertia'] for k in k_values_q2], 'bo-', label='Original (Q2)')
    plt.xlabel('Número de Clusters (K)')
    plt.ylabel('Inércia')
    plt.title('Método do Cotovelo - Dados Originais\n(Questão 2)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 2: PCA comparação
    plt.subplot(1, 3, 2)
    colors = ['red', 'green', 'purple']
    markers = ['o-', 's-', '^-']
    for i, n_comp in enumerate(sorted(results.keys())):
        result = results[n_comp]
        k_results = result['k_results']
        inertias = [k_results[k]['inertia'] for k in k_values_q2]
        plt.plot(k_values_q2, inertias, markers[i], color=colors[i], label=f'PCA-{n_comp}')
    
    plt.xlabel('Número de Clusters (K)')
    plt.ylabel('Inércia')
    plt.title('Método do Cotovelo - PCA')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 3: Comparação lado a lado do melhor K
    plt.subplot(1, 3, 3)
    methods = ['Original']
    inertias_comparison = [best_inertia_original]
    
    for n_comp in sorted(results.keys()):
        methods.append(f'PCA-{n_comp}')
        inertias_comparison.append(results[n_comp]['best_inertia'])
    
    bars = plt.bar(methods, inertias_comparison, color=['blue'] + ['red', 'green', 'purple'][:len(results)])
    plt.xlabel('Método')
    plt.ylabel('Inércia (Melhor K)')
    plt.title('Comparação de Performance\n(Melhor K por método)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Adicionar valores nas barras
    for bar, value in zip(bars, inertias_comparison):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(inertias_comparison)*0.01,
                f'{value:.0f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('comparacao_questao2_vs_pca.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("📊 Gráfico de comparação salvo: comparacao_questao2_vs_pca.png")
    
    print(f"\n=== 📈 VISUALIZAÇÃO 2D ===")
    print("🔄 Criando visualização PCA 2D...")
    
    pca_2d = PCA(n_components=2, random_state=42)
    X_2d = pca_2d.fit_transform(X_original)
    
    # K-means em 2D com melhor K encontrado
    kmeans_2d = KMeans(n_clusters=best_k_original, random_state=42)
    labels_2d = kmeans_2d.fit_predict(X_2d)
    
    # Plot otimizado
    plt.figure(figsize=(10, 6))
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    plot_sample_size = min(5000, len(X_2d))
    if len(X_2d) > plot_sample_size:
        idx = np.random.choice(len(X_2d), plot_sample_size, replace=False)
        X_2d_plot = X_2d[idx]
        labels_2d_plot = labels_2d[idx]
    else:
        X_2d_plot = X_2d
        labels_2d_plot = labels_2d
    
    for i in range(best_k_original):
        mask = labels_2d_plot == i
        if np.any(mask):
            plt.scatter(X_2d_plot[mask, 0], X_2d_plot[mask, 1], 
                       c=colors[i % len(colors)], label=f'Cluster {i}', alpha=0.6, s=20)
    
    var1, var2 = pca_2d.explained_variance_ratio_
    plt.xlabel(f'PC1 ({var1:.3f} var)')
    plt.ylabel(f'PC2 ({var2:.3f} var)')
    plt.title(f'K-means com PCA 2D - K={best_k_original} (Melhor da Questão 2)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('pca_2d_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("📊 Visualização 2D salva: pca_2d_visualization.png")
    
    # Escolher melhor resultado PCA baseado na melhoria de inércia
    best_pca_n_comp = None
    best_improvement = -float('inf')
    
    for n_comp in results.keys():
        result = results[n_comp]
        improvement = (best_inertia_original - result['best_inertia']) / best_inertia_original
        if improvement > best_improvement:
            best_improvement = improvement
            best_pca_n_comp = n_comp
    
    if best_pca_n_comp:
        print(f"\n=== ANÁLISE FINAL - COMPARAÇÃO QUESTÃO 2 vs MELHOR PCA ===")
        best_pca_result = results[best_pca_n_comp]
        
        # Adicionar colunas de cluster ao dataframe
        df['cluster_original_q2'] = best_labels_original
        df['cluster_pca'] = best_pca_result['best_labels']
        
        output_file = 'resultados_comparacao_q2_vs_pca.csv'
        df.to_csv(output_file, index=False)
        print(f"💾 Resultados comparativos salvos: {output_file}")
        
        # Estatísticas finais comparativas
        print(f"\n🔍 RESUMO COMPARATIVO:")
        print(f"{'Método':<20} {'K':<3} {'Dimensões':<10} {'Inércia':<10} {'Variância':<10}")
        print("-" * 60)
        print(f"{'Questão 2 (Original)':<20} {best_k_original:<3} {X_original.shape[1]:<10} {best_inertia_original:<10.2f} {'1.000':<10}")
        print(f"{'PCA Melhor':<20} {best_pca_result['best_k']:<3} {best_pca_n_comp:<10} {best_pca_result['best_inertia']:<10.2f} {best_pca_result['explained_variance']:<10.3f}")
        
        improvement_percent = best_improvement * 100
        print(f"\n🎯 CONCLUSÕES:")
        print(f"   • Melhor PCA: {best_pca_n_comp} componentes")
        print(f"   • Redução dimensional: {X_original.shape[1]} → {best_pca_n_comp} ({(1-best_pca_n_comp/X_original.shape[1])*100:.0f}% redução)")
        print(f"   • Variância preservada: {best_pca_result['explained_variance']:.3f}")
        print(f"   • Melhoria na inércia: {improvement_percent:+.2f}%")
        
        if improvement_percent > 0:
            print(f"   ✅ PCA MELHOROU o clustering!")
        else:
            print(f"   ⚠️  PCA teve trade-off: perdeu qualidade mas ganhou eficiência")
        
        print(f"\n⚡ Análise comparativa Questão 2 vs PCA concluída!")
        print(f"   Dataset: {len(df)} produtos")
        print(f"   Arquivos gerados: {output_file}, comparacao_questao2_vs_pca.png, pca_2d_visualization.png")
    
    else:
        print("❌ Nenhum resultado PCA válido encontrado para comparação")

if __name__ == "__main__":
    main()
