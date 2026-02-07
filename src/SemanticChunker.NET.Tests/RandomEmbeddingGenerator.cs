using Microsoft.Extensions.AI;

namespace SemanticChunkerNET.Tests;

/// <summary>
/// Simple random embedding generator for testing purposes.
/// Generates random embeddings to avoid external dependencies.
/// </summary>
public class RandomEmbeddingGenerator : IEmbeddingGenerator<string, Embedding<float>>
{
    private readonly int _embeddingDimension;

    public RandomEmbeddingGenerator(int embeddingDimension = 384)
    {
        _embeddingDimension = embeddingDimension;
    }

    public EmbeddingGeneratorMetadata Metadata => new("random-embedding-generator");

    public Task<GeneratedEmbeddings<Embedding<float>>> GenerateAsync(
        IEnumerable<string> values, 
        EmbeddingGenerationOptions? options = null, 
        CancellationToken cancellationToken = default)
    {
        var embeddings = values.Select(value => GenerateEmbedding(value)).ToList();
        return Task.FromResult(new GeneratedEmbeddings<Embedding<float>>(embeddings));
    }

    public Task<Embedding<float>> GenerateAsync(
        string value, 
        EmbeddingGenerationOptions? options = null, 
        CancellationToken cancellationToken = default)
    {
        return Task.FromResult(GenerateEmbedding(value));
    }

    private Embedding<float> GenerateEmbedding(string value)
    {
        // Generate deterministic random embeddings based on the string hash
        var hash = value.GetHashCode();
        var localRandom = new Random(hash);
        
        var vector = new float[_embeddingDimension];
        for (int i = 0; i < _embeddingDimension; i++)
        {
            vector[i] = (float)localRandom.NextDouble();
        }

        return new Embedding<float>(vector);
    }

    public object? GetService(Type serviceType, object? serviceKey = null)
    {
        return null;
    }

    public void Dispose()
    {
        // No resources to dispose
    }
}
