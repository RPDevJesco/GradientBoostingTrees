namespace GradientBoostingTrees
{
    public class MeanSquaredError : ILossFunction
    {
        public double ComputeGradient(double actual, double predicted)
        {
            return 2 * (predicted - actual);
        }

        public double ComputeLoss(double actual, double predicted)
        {
            return Math.Pow(predicted - actual, 2);
        }
    }
}