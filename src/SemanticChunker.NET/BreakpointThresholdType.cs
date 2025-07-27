namespace SemanticChunkerNET;

/// <summary>
/// Controls how the breakpoint threshold is calculated.
/// </summary>
public enum BreakpointThresholdType
{
    Percentile,
    StandardDeviation,
    InterQuartile,
    Gradient
}