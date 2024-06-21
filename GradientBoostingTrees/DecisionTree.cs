namespace GradientBoostingTrees
{
public class DecisionTree : ITree
    {
        private Node _root;
        private readonly int _maxDepth;

        public DecisionTree(int maxDepth = 3)
        {
            _maxDepth = maxDepth;
        }

        public void Fit(double[][] features, double[] targets)
        {
            _root = BuildTree(features, targets, 0);
        }

        public double Predict(double[] features)
        {
            return Predict(features, _root);
        }

        private Node BuildTree(double[][] features, double[] targets, int depth)
        {
            if (depth >= _maxDepth || targets.Length <= 1)
            {
                return new Node { IsLeaf = true, Value = targets.Length == 0 ? 0 : targets.Average() };
            }

            int bestFeature = 0;
            double bestThreshold = 0;
            double bestImpurity = double.MaxValue;

            for (int featureIndex = 0; featureIndex < features[0].Length; featureIndex++)
            {
                var sortedIndices = features
                    .Select((f, i) => new { Feature = f[featureIndex], Index = i })
                    .OrderBy(f => f.Feature)
                    .Select(f => f.Index)
                    .ToArray();

                for (int i = 1; i < sortedIndices.Length; i++)
                {
                    double threshold = (features[sortedIndices[i - 1]][featureIndex] + features[sortedIndices[i]][featureIndex]) / 2;
                    var (leftSplitTargets, rightSplitTargets) = Split(targets, sortedIndices, i);

                    double impurity = ComputeImpurity(leftSplitTargets, rightSplitTargets);
                    if (impurity < bestImpurity)
                    {
                        bestImpurity = impurity;
                        bestFeature = featureIndex;
                        bestThreshold = threshold;
                    }
                }
            }

            var (leftFeatures, leftTargets, rightFeatures, rightTargets) = Split(features, targets, bestFeature, bestThreshold);

            return new Node
            {
                FeatureIndex = bestFeature,
                Threshold = bestThreshold,
                Left = BuildTree(leftFeatures, leftTargets, depth + 1),
                Right = BuildTree(rightFeatures, rightTargets, depth + 1)
            };
        }

        private double Predict(double[] features, Node node)
        {
            if (node.IsLeaf)
            {
                return node.Value;
            }

            if (features[node.FeatureIndex] <= node.Threshold)
            {
                return Predict(features, node.Left);
            }
            else
            {
                return Predict(features, node.Right);
            }
        }

        private (double[] leftSplitTargets, double[] rightSplitTargets) Split(double[] targets, int[] sortedIndices, int splitIndex)
        {
            return (
                sortedIndices.Take(splitIndex).Select(i => targets[i]).ToArray(),
                sortedIndices.Skip(splitIndex).Select(i => targets[i]).ToArray()
            );
        }

        private (double[][] leftFeatures, double[] leftTargets, double[][] rightFeatures, double[] rightTargets) Split(double[][] features, double[] targets, int featureIndex, double threshold)
        {
            var leftFeatures = new List<double[]>();
            var leftTargets = new List<double>();
            var rightFeatures = new List<double[]>();
            var rightTargets = new List<double>();

            for (int i = 0; i < features.Length; i++)
            {
                if (features[i][featureIndex] <= threshold)
                {
                    leftFeatures.Add(features[i]);
                    leftTargets.Add(targets[i]);
                }
                else
                {
                    rightFeatures.Add(features[i]);
                    rightTargets.Add(targets[i]);
                }
            }

            return (leftFeatures.ToArray(), leftTargets.ToArray(), rightFeatures.ToArray(), rightTargets.ToArray());
        }

        private double ComputeImpurity(double[] leftTargets, double[] rightTargets)
        {
            double leftImpurity = leftTargets.Length == 0 ? 0 : leftTargets.Select(t => Math.Pow(t - leftTargets.Average(), 2)).Average();
            double rightImpurity = rightTargets.Length == 0 ? 0 : rightTargets.Select(t => Math.Pow(t - rightTargets.Average(), 2)).Average();
            return (leftTargets.Length * leftImpurity + rightTargets.Length * rightImpurity) / (leftTargets.Length + rightTargets.Length);
        }
    }
}