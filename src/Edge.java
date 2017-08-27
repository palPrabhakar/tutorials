public class Edge implements Comparable<Edge> {
	private int v, w;
	private double weight;
	
	public Edge(int v, int w, double weight) {
		this.v = v;
		this.w = w;
		this.weight = weight;
	}
	
	public double weight() {
		return weight;
	}
	
	public int either() {
		return v;
	}
	
	public int other(int v) {
		if(v == this.v) return this.w;
		return this.v;
	}
	
	@Override
	public int compareTo(Edge that) {
		// TODO Auto-generated method stub
		if(this.weight() < that.weight()) return -1;
		else if(this.weight() == that.weight()) return 0;
		else return 1;
	}
	
	public String toString() {
		return String.format("%d %d %.2f", v, w, weight);
	}
}
