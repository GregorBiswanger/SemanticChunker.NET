using Microsoft.Extensions.AI;

namespace SemanticChunkerNET.Tests;

/// <summary>
/// Simple random embedding generator for testing purposes.
/// Generates random embeddings to avoid external dependencies.
/// </summary>
public class RandomEmbeddingGenerator : IEmbeddingGenerator<string, Embedding<float>>
{
    private readonly Random _random;
    private readonly int _embeddingDimension;

    public RandomEmbeddingGenerator(int seed = 42, int embeddingDimension = 384)
    {
        _random = new Random(seed);
        _embeddingDimension = embeddingDimension;
    }

    public EmbeddingGeneratorMetadata Metadata => new("random-embedding-generator");

    public async Task<GeneratedEmbeddings<Embedding<float>>> GenerateAsync(
        IEnumerable<string> values, 
        EmbeddingGenerationOptions? options = null, 
        CancellationToken cancellationToken = default)
    {
        var embeddings = new List<Embedding<float>>();
        
        foreach (var value in values)
        {
            embeddings.Add(await GenerateAsync(value, options, cancellationToken));
        }

        return new GeneratedEmbeddings<Embedding<float>>(embeddings);
    }

    public Task<Embedding<float>> GenerateAsync(
        string value, 
        EmbeddingGenerationOptions? options = null, 
        CancellationToken cancellationToken = default)
    {
        // Generate deterministic random embeddings based on the string hash
        var hash = value.GetHashCode();
        var localRandom = new Random(hash);
        
        var vector = new float[_embeddingDimension];
        for (int i = 0; i < _embeddingDimension; i++)
        {
            vector[i] = (float)localRandom.NextDouble();
        }

        var embedding = new Embedding<float>(vector);
        return Task.FromResult(embedding);
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
