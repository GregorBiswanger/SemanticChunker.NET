using ICU4N.Text;

namespace SemanticChunkerNET;

/// <summary>
/// Provides methods for splitting text into sentences, building contextual
/// sentence windows, and splitting oversized chunks at natural boundaries.
/// </summary>
internal static class TextSegmenter
{
    /// <summary>
    /// Splits <paramref name="text"/> into individual sentences using the
    /// ICU4N sentence break iterator.
    /// </summary>
    internal static IList<string> SplitIntoSentences(string text)
    {
        var result = new List<string>();
        BreakIterator iterator = BreakIterator.GetSentenceInstance();
        iterator.SetText(text);

        for (int start = iterator.First(), end = iterator.Next();
             end != BreakIterator.Done;
             start = end, end = iterator.Next())
        {
            string sentence = text.Substring(start, end - start).Trim();
            if (sentence.Length > 0)
            {
                result.Add(sentence);
            }
        }

        return result;
    }

    /// <summary>
    /// Builds contextual sentences by combining each sentence with up to
    /// <paramref name="buffer"/> surrounding sentences on each side.
    /// </summary>
    internal static IList<string> BuildContextualSentences(IList<string> sentences, int buffer)
    {
        var result = new List<string>(sentences.Count);

        // Clamp buffer to 0 to ensure current sentence is always included
        buffer = Math.Max(0, buffer);

        for (int i = 0; i < sentences.Count; i++)
        {
            var startInclusive = Math.Max(0, i - buffer);
            var endExclusive = Math.Min(i + buffer + 1, sentences.Count);
            var context = sentences.Skip(startInclusive).Take(endExclusive - startInclusive);

            result.Add(string.Join(' ', context));
        }

        return result;
    }

    /// <summary>
    /// Splits <paramref name="text"/> into parts that each fit within
    /// <paramref name="maxChars"/>, preferring newline boundaries within
    /// <paramref name="overrun"/> characters before falling back to a hard cut.
    /// </summary>
    internal static IEnumerable<string> SplitChunkText(string text, int maxChars, int overrun)
    {
        while (text.Length > maxChars)
        {
            int cutIndex = maxChars;

            if (overrun > 0)
            {
                int searchEnd = Math.Min(text.Length, maxChars + overrun);
                int newlineIndex = text.IndexOf('\n', maxChars, searchEnd - maxChars);

                if (newlineIndex >= 0)
                {
                    cutIndex = newlineIndex;
                }
            }

            yield return text[..cutIndex];
            text = text[cutIndex..].TrimStart('\n');
        }

        if (text.Length > 0)
        {
            yield return text;
        }
    }
}
