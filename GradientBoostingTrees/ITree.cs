namespace GradientBoostingTrees
{
    public interface ITree
    {
        double Predict(double[] features);
    }
}