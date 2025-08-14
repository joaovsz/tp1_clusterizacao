"""
Questão 2: Aplicação simples do K-means no dataset da ShopMania
"""

import kagglehub
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import os

def main():
    print("=== QUESTÃO 2: K-MEANS NO DATASET DA SHOPMANIA ===\n")
    
    # 1. Download do dataset
    print("Baixando dataset da ShopMania...")
    path = kagglehub.dataset_download("lakritidis/product-classification-and-categorization")
    print(f"Path to dataset files: {path}")
    
    # 2. Carregar dados
    files = os.listdir(path)
    csv_files = [f for f in files if f.endswith('.csv')]
    print(f"Arquivos CSV encontrados: {csv_files}")
    
    # Tentar carregar diferentes arquivos CSV
    for csv_file in csv_files:
        print(f"\n--- Testando arquivo: {csv_file} ---")
        try:
            df = pd.read_csv(os.path.join(path, csv_file))
            print(f"Shape: {df.shape}")
            print(f"Colunas: {list(df.columns)}")
            
            # Usar o arquivo shopmania.csv se existir, senão o primeiro
            if 'shopmania' in csv_file.lower():
                break
        except Exception as e:
            print(f"Erro ao ler {csv_file}: {e}")
            continue
    
    print(f"\n--- USANDO ARQUIVO: {csv_file} ---")
    print(f"Dataset carregado: {df.shape[0]} linhas, {df.shape[1]} colunas")
    print(f"Colunas: {list(df.columns)}")
    print("\nPrimeiras 5 linhas:")
    print(df.head())
    
    # Verificar se há coluna 'category' para análise
    if 'category' in [col.lower() for col in df.columns]:
        cat_col = [col for col in df.columns if col.lower() == 'category'][0]
        print(f"\nCategorias únicas: {df[cat_col].nunique()}")
        print(f"Top 10 categorias:")
        print(df[cat_col].value_counts().head(10))
    
    # 3. Preparar dados para clustering
    print("\n=== PREPARANDO DADOS ===")
    
    # Selecionar coluna de texto mais apropriada
    text_columns = [col for col in df.columns if df[col].dtype == 'object']
    print(f"Colunas de texto encontradas: {text_columns}")
    
    # Tentar identificar coluna de produto/nome
    product_col = None
    possible_names = ['product', 'name', 'title', 'description']
    
    for col in text_columns:
        col_lower = col.lower()
        if any(name in col_lower for name in possible_names):
            product_col = col
            break
    
    # Se não encontrou, usar a coluna com mais texto único
    if not product_col and text_columns:
        # Escolher coluna com maior diversidade de texto
        text_diversity = {}
        for col in text_columns[:3]:  # Testar apenas primeiras 3 colunas
            if col.lower() != 'category':
                unique_ratio = df[col].nunique() / len(df)
                text_diversity[col] = unique_ratio
        
        if text_diversity:
            product_col = max(text_diversity, key=text_diversity.get)
            print(f"Coluna escolhida (maior diversidade): '{product_col}'")
    
    if product_col:
        print(f"Usando coluna '{product_col}' para clustering")
        
        # Preencher valores nulos e limpar texto
        text_data = df[product_col].fillna('').astype(str)
        
        # Amostrar dados se muito grande (para teste mais rápido)
        if len(text_data) > 50000:
            print(f"Dataset grande ({len(text_data)} linhas). Usando amostra de 50,000 para análise mais rápida...")
            sample_idx = np.random.choice(len(text_data), 50000, replace=False)
            text_data = text_data.iloc[sample_idx]
            df = df.iloc[sample_idx].reset_index(drop=True)
            print(f"Nova amostra: {len(text_data)} produtos")
        
        # Aplicar TF-IDF (removendo stopwords em múltiplas línguas)
        vectorizer = TfidfVectorizer(
            max_features=1000, 
            stop_words='english',  # Pode adicionar grego também
            min_df=2,  # Palavras devem aparecer pelo menos 2 vezes
            max_df=0.95,  # Remover palavras muito comuns
            ngram_range=(1, 2)  # Uni e bigramas
        )
        X = vectorizer.fit_transform(text_data)
        X = X.toarray()
        
        print(f"Matriz TF-IDF criada: {X.shape}")
        print(f"Exemplos de features: {vectorizer.get_feature_names_out()[:10]}")
    else:
        print("Nenhuma coluna de texto apropriada encontrada!")
        return
    
    # 4. Aplicar K-means
    print("\n=== APLICANDO K-MEANS ===")
    
    # Testar alguns valores de k
    k_values = [3, 5, 8, 10]
    results = {}
    
    for k in k_values:
        print(f"\nTestando k={k}")
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        
        # Calcular inércia
        inertia = kmeans.inertia_
        results[k] = {'labels': labels, 'inertia': inertia, 'model': kmeans}
        
        print(f"Inércia: {inertia:.2f}")
        
        # Mostrar distribuição dos clusters
        unique, counts = np.unique(labels, return_counts=True)
        for cluster_id, count in zip(unique, counts):
            print(f"  Cluster {cluster_id}: {count} produtos")
    
    # 5. Plotar método do cotovelo
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, [results[k]['inertia'] for k in k_values], 'bo-')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Inércia')
    plt.title('Método do Cotovelo')
    plt.grid(True)
    plt.show()
    
    # 6. Escolher melhor k (usar método do cotovelo) e analisar
    # Calcular diferenças para encontrar "cotovelo"
    inertias = [results[k]['inertia'] for k in k_values]
    
    # Método simples do cotovelo: maior queda relativa
    if len(inertias) > 2:
        drops = []
        for i in range(1, len(inertias)):
            drop = (inertias[i-1] - inertias[i]) / inertias[i-1]
            drops.append(drop)
        best_k_idx = drops.index(max(drops))
        best_k = k_values[best_k_idx + 1]
    else:
        best_k = k_values[1] if len(k_values) > 1 else k_values[0]
    
    print(f"\nMelhor k escolhido pelo método do cotovelo: {best_k}")
    
    best_labels = results[best_k]['labels']
    
    print(f"\n=== ANÁLISE DETALHADA COM K={best_k} ===")
    
    # Adicionar clusters ao dataframe
    df['cluster'] = best_labels
    
    # Análise mais detalhada dos clusters
    for cluster_id in range(best_k):
        cluster_data = df[df['cluster'] == cluster_id]
        print(f"\n--- Cluster {cluster_id} ({len(cluster_data)} produtos, {len(cluster_data)/len(df)*100:.1f}%) ---")
        
        # Mostrar alguns produtos deste cluster
        sample_products = cluster_data[product_col].head(5).tolist()
        print("Exemplos de produtos:")
        for i, product in enumerate(sample_products, 1):
            print(f"  {i}. {product[:100]}...")  # Truncar texto longo
        
        # Se há categorias verdadeiras, mostrar distribuição
        category_cols = [col for col in df.columns if 'categ' in col.lower()]
        if category_cols:
            cat_col = category_cols[0]
            top_categories = cluster_data[cat_col].value_counts().head(5)
            print("Categorias mais comuns:")
            for cat, count in top_categories.items():
                percentage = count / len(cluster_data) * 100
                print(f"    - {cat}: {count} ({percentage:.1f}%)")
        
        # Palavras mais importantes neste cluster
        cluster_indices = np.where(best_labels == cluster_id)[0]
        if len(cluster_indices) > 0:
            cluster_vectors = X[cluster_indices]
            mean_vector = cluster_vectors.mean(axis=0)
            top_features_idx = mean_vector.argsort()[-10:][::-1]  # Top 10 features
            feature_names = vectorizer.get_feature_names_out()
            
            print("Palavras-chave mais importantes:")
            top_words = [feature_names[i] for i in top_features_idx]
            print(f"    {', '.join(top_words)}")
    
    # Estatísticas gerais
    print(f"\n=== ESTATÍSTICAS GERAIS ===")
    cluster_sizes = df['cluster'].value_counts().sort_index()
    print("Distribuição dos clusters:")
    for cluster_id, size in cluster_sizes.items():
        print(f"  Cluster {cluster_id}: {size} produtos ({size/len(df)*100:.1f}%)")
    
    # Se há categorias, mostrar pureza dos clusters
    if category_cols:
        cat_col = category_cols[0]
        print(f"\nPureza dos clusters (baseada na categoria '{cat_col}'):")
        for cluster_id in range(best_k):
            cluster_data = df[df['cluster'] == cluster_id]
            if len(cluster_data) > 0:
                most_common_cat = cluster_data[cat_col].value_counts().iloc[0]
                purity = most_common_cat / len(cluster_data)
                print(f"  Cluster {cluster_id}: {purity:.2f} ({most_common_cat}/{len(cluster_data)} produtos da mesma categoria)")
    
    # 7. Salvar resultados
    df.to_csv('resultados_kmeans_simples.csv', index=False)
    print(f"\nResultados salvos em 'resultados_kmeans_simples.csv'")
    
    print(f"\n=== RESUMO FINAL ===")
    print(f"📊 Dataset: {df.shape[0]} produtos analisados")
    print(f"🔢 Features: {X.shape[1]} características TF-IDF extraídas")
    print(f"📈 Clusters: {best_k} grupos identificados")
    print(f"📉 Inércia final: {results[best_k]['inertia']:.2f}")
    print(f"🎯 Interpretação: O K-means conseguiu agrupar produtos por tipo/marca")
    if category_cols:
        avg_purity = df.groupby('cluster')[cat_col].apply(lambda x: x.value_counts().iloc[0] / len(x)).mean()
        print(f"✅ Pureza média dos clusters: {avg_purity:.2f} (0=ruim, 1=perfeito)")

if __name__ == "__main__":
    main()
