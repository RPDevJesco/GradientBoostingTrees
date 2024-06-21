namespace GradientBoostingTrees
{
    public class Node
    {
        public bool IsLeaf { get; set; }
        public double Value { get; set; }
        public int FeatureIndex { get; set; }
        public double Threshold { get; set; }
        public Node Left { get; set; }
        public Node Right { get; set; }
    }
}