import edu.princeton.cs.algs4.MinPQ;
import edu.princeton.cs.algs4.Queue;

public class LazyPrimMST {
	
	private MinPQ<Edge> pq;
	private boolean[] marked;
	private Queue<Edge> mst; 
	private double weight;
	
	public LazyPrimMST(EdgeWeighterGraph g) {
		//System.out.printf("Debug : %d\n\n", g.v());
		marked = new boolean[g.v()];
		pq = new MinPQ<Edge>();
		mst = new Queue<Edge>();
		
		visit(g, 0);
		while (!pq.isEmpty() || mst.size() != g.v() - 1) {
			Edge e = pq.delMin();
			int v = e.either();
			int w = e.other(v);
			
			if (!marked[v]) { 
				weight(e);
				mst.enqueue(e);
				visit(g, v);
			}
			if (!marked[w]) {
				weight(e);
				mst.enqueue(e);
				visit(g, w);
			}
		}
		
	}
	
	public double weight(){
		return this.weight;
	}
	
	private void weight(Edge e) {
		this.weight += e.weight();
	}
	
	private void visit(EdgeWeighterGraph g, int v) {
		marked[v] = true;
		for (Edge e : g.adj(v)) {
			int w = e.other(v);
			//System.out.printf("%b : %d\n", !marked[w], w);
			if(!marked[w]) 
				pq.insert(e);
		}
	}
	
	public Iterable<Edge> mst() {
		return mst;
	}
}
