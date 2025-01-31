public class DirectedEdge implements Comparable<Edge> {
	private int v;
	private int w;
	private double weight;
	
	public DirectedEdge(int v, int w, double weight) {
		this.v = v;
		this.w = w;
		this.weight = weight;
	}
	
	public double weight() {
		return this.weight;
	}
	
	public int from() {
		return this.v;
	}
	
	public int to() {
		return this.w;
	}
	
	public String toString() {
		return String.format("%d %d %.2f", this.v, this.w, this.weight);
	}
	
	public int compareTo(Edge that) {
		if(this.weight() < that.weight()) return -1;
		else if(this.weight() == that.weight()) return 0;
		else return 1;
	}
}
