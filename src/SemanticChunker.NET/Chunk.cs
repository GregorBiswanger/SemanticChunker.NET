using Microsoft.Extensions.AI;

namespace SemanticChunkerNET;

public class Chunk
{
    public required string Id { get; set; }
    public required string Text { get; set; }
    public required Embedding<float> Embedding { get; set; }
}