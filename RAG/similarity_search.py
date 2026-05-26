from RAG.embedder import embed_image
from RAG.db_conn import get_collection

collection = get_collection()


def find_similar_images(
    query_image_path: str,
    k: int = 3,
    dataset: str = None
):

    query_embedding = embed_image(query_image_path)

    where_filter = None

    if dataset:
        where_filter = {
            "dataset": dataset
        }

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=k,
        where=where_filter
    )

    output = []

    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for meta, dist in zip(metadatas, distances):

        output.append(
            (
                meta["image_path"],
                meta["dataset"],
                float(dist)
            )
        )

    return output


if __name__ == "__main__":

    query_image = "C:\\NeuroVisionCombined4\\data\\gopro_deblur\\sharp\\images\\000001.png"

    results = find_similar_images(
        query_image,
        k=3
    )

    for i, (path, ds, dist) in enumerate(results, 1):

        print(
            f"{i}. {path} | {ds} | distance={dist:.4f}"
        )