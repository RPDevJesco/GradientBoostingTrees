namespace GradientBoostingTrees
{
    public interface ILossFunction
    {
        double ComputeGradient(double actual, double predicted);
        double ComputeLoss(double actual, double predicted);
    }
}