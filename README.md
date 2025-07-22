[![SemanticChunker.NET Logo](https://github.com/GregorBiswanger/SemanticChunker.NET/raw/main/assets/semantic-chunker-net-logo-transparent.png)](https://github.com/GregorBiswanger/SemanticChunker.NET)  

# SemanticChunker.NET

**Automatic Semantic Chunking for RAG in .NET  
Transforms long texts into coherent, retrieval ready chunks with a single call - powered by embeddings and fully compatible with Semantic Kernel and Microsoft.Extensions.AI.**

[![NuGet](https://img.shields.io/nuget/v/SemanticChunker.NET?style=flat-square)](https://www.nuget.org/packages/SemanticChunker.NET/)
![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)

> Split long documents into semantically coherent chunks that fit your LLM’s context window while maximising retrieval precision.

## Features ✨

- **Plug‑and‑play API** – One call to `CreateChunksAsync` returns ready‑to‑use `Chunk` objects with ID, text, and embedding.
- **Model‑agnostic** – Works with any embedding generator supported by `Microsoft.Extensions.AI`; no framework lock‑in.
- **Four breakpoint strategies** – `Percentile`, `StandardDeviation`, `InterQuartile`, and `Gradient` cover most corpus profiles.
- **Context buffer window** – Configurable `bufferSize` preserves cross‑sentence semantics.
- **Target chunk count** – Unique `targetChunkCount` option produces exactly the number of chunks you need.
- **Multilingual sentence splitting** – ICU4N ensures accurate sentence boundaries in 70+ languages.
- **Token‑limit safety** – Automatic 10 % safety margin below your model’s context window.
- **Parallel embedding generation** – Maximises throughput when your embedding provider supports batching.
- **Zero external overhead** – Pure .NET plus ICU4N; lightweight for microservices and serverless functions.

## Installation 📦

```bash
dotnet add package SemanticChunker.NET
````

## Quick Start 🛠️

```csharp
using Microsoft.Extensions.AI;
using Microsoft.SemanticKernel;
using SemanticChunker.NET;

// 1. Wire an embedding generator (example uses LM Studio)
var builder = Kernel.CreateBuilder();
builder.Services.AddLmStudioEmbeddingGenerator("text-embedding-multilingual-e5-base");
using var kernel = builder.Build();

// 2. Create Chunker with your model’s token limit (e.g. 512)
var embeddingGenerator =
    kernel.Services.GetRequiredService<IEmbeddingGenerator<string, Embedding<float>>>();

var chunker = new SemanticChunker(embeddingGenerator, tokenLimit: 512);

// 3. Chunk text
string input = File.ReadAllText("whitepaper.md");
IList<Chunk> chunks = await chunker.CreateChunksAsync(input);

// 4. Persist embeddings to your vector store
await myVectorStore.UpsertAsync(chunks);
```

## Step‑by‑Step Calibration Guide

This section walks you through finding the *best* settings for **your** corpus and embedding model.

| Step                                    | Action                                                                                                                                            | Why                                                                        |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| **1 Choose an embedding model**         | Prefer models whose training data match your language/domain. | Embedding quality dominates chunk quality.                    |
| **2 Set `tokenLimit`**                  | Use the embedding model token limit.                                                                        | Leaves headroom for prompts/RAG metadata.            |
| **3 Pick a buffer size**                | Start with **1**; raise to 2–3 if individual sentences lose context.                                                                              | Neighbor sentences improve semantic continuity.               |
| **4 Choose a breakpoint strategy**      | `Percentile` 95 % is the industry default. Switch to `StandardDeviation` when your corpus shows heavy-tail distance distributions.                | Percentile is robust; SD handles outliers.          |
| **5 Adjust `thresholdAmount`**          | Lower value → more chunks, higher recall; Higher value → fewer, longer chunks, better precision. Tune in 5‑point increments (e.g. 90, 95, 98).    | Balances retrieval recall vs. answer accuracy.                |
| **6 Optionally set `targetChunkCount`** | If you know how many chunks you need (e.g. for fixed‑budget eval), supply it and skip manual threshold tuning.                                    | Directly controls output size.                                             |
| **7 Evaluate**                          | Measure Answer F1/EM and retrieval hit rate on a validation set. Iterate Steps 4–6 until metrics plateau.                                         | Empirical tuning beats rules of thumb.  |
| **8 Lock parameters in production**     | Persist calibrated values in app settings or environment variables.                                                                               | Guarantees reproducibility across builds.                                  |

## Configuration Reference

| Ctor Parameter     | Default      | Description                                                           |
| ------------------ | ------------ | --------------------------------------------------------------------- |
| `tokenLimit`       | *‑*          | Max LLM tokens per chunk (safety margin = 10 %).                      |
| `bufferSize`       | `1`          | Sentences added before/after current sentence during embedding.       |
| `thresholdType`    | `Percentile` | Breakpoint metric (`StandardDeviation`, `InterQuartile`, `Gradient`). |
| `thresholdAmount`  | see table    | E.g. 95 % for Percentile, 3 σ for Standard Deviation.                 |
| `targetChunkCount` | `null`       | Overrides thresholds to hit an exact chunk count.                     |
| `minChunkChars`    | `0`          | Skip chunks shorter than this.                                        |

## 👨‍💻 Author

**[Gregor Biswanger](https://github.com/GregorBiswanger)** - is a leading expert in generative AI, a Microsoft MVP for Azure AI and Web App Development. As an independent consultant, he works closely with the Microsoft product team for GitHub Copilot and supports companies in implementing modern AI solutions.

 As a freelance consultant, trainer, and author, he shares his expertise in software architecture and cloud technologies and is a sought-after speaker at international conferences. For several years, he has been live-streaming every Friday evening on [Twitch](https://twitch.tv/GregorBiswanger) with [My Coding Zone](https://www.my-coding-zone.de) in german and is an active [YouTuber](https://youtube.com/GregorBiswanger).

Reach out to Gregor if you need support in the form of consulting, training, or implementing AI solutions using .NET or Node.js. [LinkedIn](https://www.linkedin.com/in/gregor-biswanger-51342011/) or Twitter [@BFreakout](https://www.twitter.com/BFreakout)  

See also the list of [contributors](https://github.com/GregorBiswanger/SemanticChunker.NET/graphs/contributors) who participated in this project.

## 🙋‍♀️🙋‍♂ Contributing

Feel free to submit a pull request if you find any bugs (to see a list of active issues, visit the [Issues section](https://github.com/GregorBiswanger/SemanticChunker.NET/issues).
Please make sure all commits are properly documented.

The best thing would be to write about what you plan to do in the issue beforehand. Then there will be no disappointment if we cannot accept your pull request.

## 🙏 Donate

I work on this open-source project in my free time alongside a full-time job and raising three kids. If you`d like to support my work and help me dedicate more time to this project, consider sponsoring me on GitHub:  

- [Gregor Biswanger](https://github.com/sponsors/GregorBiswanger)  

Your sponsorship allows me to invest more time in improving the project and prioritizing important issues or features. Any support is greatly appreciated - thank you! 🍻  

## 📜 License

This project is licensed under the [**Apache License 2.0**](https://raw.githubusercontent.com/GregorBiswanger/SemanticChunker.NET/refs/heads/main/LICENSE.txt) - © Gregor Biswanger 2025

*Happy chunking!* 🧩
