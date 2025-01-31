import edu.princeton.cs.algs4.*;

public class EdgeWeightedDigraph {
	private Bag<DirectedEdge>[] adj;
	private int v;
	private int e;
	
	public EdgeWeightedDigraph(int v) {
		this.v = v;
		adj = (Bag<DirectedEdge>[]) new Bag[v];
		for(int i = 0; i < v; i++) {
			adj[i] = new Bag<DirectedEdge>();
		}
	}
	
	public EdgeWeightedDigraph(In in) {
		this(in.readInt());
		int e = in.readInt();
		for(int i = 0; i < e; i++) {
			DirectedEdge _e = new DirectedEdge(in.readInt(), in.readInt(), in.readDouble());
			addEdge(_e);
		}
	}
	
	public void addEdge(DirectedEdge e) {
		adj[e.from()].add(e);
		this.e++;
	}
	
	public int v() {
		return this.v;
	}
	
	public int e() {
		return this.e;
	}
	
	public Iterable<DirectedEdge> adj(int v) {
		return adj[v];
	}
	
	
	public Iterable<DirectedEdge> edges() {
		Queue<DirectedEdge> q = new Queue<DirectedEdge>();
		
		for(int i = 0; i < this.v; i++) {
			for(DirectedEdge e : adj[i]){
				q.enqueue(e);
			}
		}
		return q;
	}
	
	public static void main(String[] args) {
		EdgeWeightedDigraph g = new EdgeWeightedDigraph(new In(args[0]));
		System.out.printf("V : %d\tE : %d\n", g.v(), g.e());
		for(DirectedEdge e : g.edges()) {
			System.out.println(e);
		}
	}
}
