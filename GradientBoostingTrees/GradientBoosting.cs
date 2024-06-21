namespace GradientBoostingTrees
{
    public class GradientBoosting
    {
        private readonly List<DecisionTree> _trees;
        private readonly ILossFunction _lossFunction;
        private readonly double _learningRate;
        private double _initialPrediction;

        public GradientBoosting(ILossFunction lossFunction, double learningRate = 0.1)
        {
            _trees = new List<DecisionTree>();
            _lossFunction = lossFunction;
            _learningRate = learningRate;
        }

        public void Fit(double[][] features, double[] targets, int numTrees, int maxDepth = 3)
        {
            _initialPrediction = targets.Average();
            double[] predictions = new double[targets.Length];
            Array.Fill(predictions, _initialPrediction);

            for (int i = 0; i < numTrees; i++)
            {
                double[] residuals = ComputeResiduals(targets, predictions);
                var tree = new DecisionTree(maxDepth);
                tree.Fit(features, residuals);
                _trees.Add(tree);

                for (int j = 0; j < predictions.Length; j++)
                {
                    predictions[j] += _learningRate * tree.Predict(features[j]);
                }
            }
        }

        public double Predict(double[] features)
        {
            double prediction = _initialPrediction;
            foreach (var tree in _trees)
            {
                prediction += _learningRate * tree.Predict(features);
            }
            return prediction;
        }

        private double[] ComputeResiduals(double[] targets, double[] predictions)
        {
            double[] residuals = new double[targets.Length];
            for (int i = 0; i < targets.Length; i++)
            {
                residuals[i] = targets[i] - predictions[i];
            }
            return residuals;
        }
    }
}