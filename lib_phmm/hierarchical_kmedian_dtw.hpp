#include<vector>
#include<cmath>
#include<cstdlib>
#include<algorithm>
#include<random>
#include<utility>
#include<set>
#include<iostream>

using namespace std;

using Vec = vector<double>; // duplicate
using Mat = vector<Vec>;    // duplicate
using Dataset = vector<Mat>;

inline double euc(Vec &x, Vec &y) {
  int n = x.size();
  double distance = 0.0;
  for(int i = 0; i < n; i++) {
    distance += pow(x[i] - y[i], 2);
  }
  return distance;
}

inline double dtw(Mat &x, Mat &y, int warping_band, bool normalize = true) {
  int n = x.size();
  int m = y.size();

  Mat dp = Mat(n + 1, Vec(m + 1, INFINITY));
  vector<vector<int>> plen(n + 1, vector<int>(m + 1, 0));
  dp[0][0] = 0.0;

  int w = max(warping_band, abs(n - m));

  for(int i = 1; i <= n; i++) {
    int j_start = max(1, i - w);
    int j_end = min(m, i + w);
    for(int j = j_start; j <= j_end; j++) {
      double up   = dp[i - 1][j];
      double diag = dp[i - 1][j - 1];
      double left = dp[i][j - 1];

      // same tie-break order as the old min3(up, diag, left)
      double best = up;
      int best_len = plen[i - 1][j];
      if(diag < best) { best = diag; best_len = plen[i - 1][j - 1]; }
      if(left < best) { best = left; best_len = plen[i][j - 1]; }

      dp[i][j] = best + euc(x[i - 1], y[j - 1]);
      plen[i][j] = best_len + 1;
    }
  }

  // normalize by warp-path length so distance is comparable across
  // candidate regions of different durations. Note the DP minimizes the
  // summed cost, not this mean, and a longer path over cheap cells lowers
  // the mean -- normalize=false returns the raw summed cost instead.
  if(!normalize) return dp[n][m];
  return dp[n][m] / plen[n][m];
}

inline int instance_pair_hash(int i, int j, int n_instances) {
  return i * n_instances + j;
}

class DistanceManagement {
public:
  DistanceManagement(Dataset *dataset, int warping_band, bool normalize = true)
    :dataset(dataset), warping_band(warping_band), normalize(normalize) {
    store = Mat(dataset->size(), Vec(dataset->size(), INFINITY));
  }

  double distance(int i, int j) {
    if(isinf(store[i][j])) {
      store[i][j] = dtw((*dataset)[i], (*dataset)[j], warping_band, normalize);
      store[j][i] = store[i][j];
    }
    return store[i][j];
  }

  double density(int anchor, vector<int> instances) {
    if(instances.empty()) {
      return 0.0;
    }
    double total = 0.0;
    for(int i = 0; i < instances.size(); i++) {
      total += this -> distance(instances[i], anchor);
    }
    double avg_distance = total / instances.size();
    if(avg_distance == 0) {
      return INFINITY;
    }
    return 1.0 / avg_distance;
  }
  
  int size() {
    return dataset -> size();
  }

private:
  int warping_band;
  bool normalize;
  Dataset *dataset;
  Mat store;
};


inline int sample_medoid(DistanceManagement &x, vector<int> &instances, int anchor, mt19937 &gen) {
  int n = instances.size();
  Vec cum_distances = Vec(n, 0.0);
  for(int i = 0; i < n; i++) {
    cum_distances[i] = x.distance(anchor, instances[i]);
    if(i > 0) cum_distances[i] += cum_distances[i - 1];
  }
  std::uniform_real_distribution<> dis(0.0, cum_distances[n - 1]);
  double r = dis(gen);
  for(int i = 0; i < n; i++) {
    if(anchor != instances[i]) {
      if(r < cum_distances[i]) return instances[i];
    }
  }
  return instances[n - 1];
}

inline pair<int, double> select_medoid(int medoid, double current_avg, vector<int> &inst, DistanceManagement &x) {
  int new_medoid = medoid;
  double min_avg_dist = current_avg;
  int n = (int)inst.size();
  for(int i = 0; i < n; i++) {
    double total = 0.0;
    for(int j = 0; j < n; j++) {
      total += x.distance(inst[i], inst[j]);
    }
    double avg = total / n;
    if(avg < min_avg_dist) {
      min_avg_dist = avg;
      new_medoid = inst[i];
    }
  }
  return make_pair(new_medoid, min_avg_dist);
}


// A single random (anchor, sample) initialization only has roughly a
// coin-flip's chance of splitting a smooth (non-sharply-clustered) real
// distance distribution into two halves that both beat the parent's
// already-optimized single-medoid density -- on real DTW-distance data
// that made branches give up (settle for one coarse medoid) after just
// 1-2 levels almost everywhere. Trying several random restarts and
// keeping the densest split makes that per-level roll far less luck-
// dependent, without reintroducing an external distance/count threshold.
const int KMEDOIDS_RESTARTS_DEFAULT = 10;

// A child only marginally less dense than its parent (e.g. from one
// restart's random assignment noise) shouldn't permanently give up on
// that whole branch -- allow it to keep splitting as long as it's within
// this fraction of the parent's density.
const double DENSITY_TOLERANCE_DEFAULT = 0.05;

inline void kmedoids(DistanceManagement &x, vector<int> &instances, set<int> &medoids, int epochs, double parent_density,
		      int restarts = KMEDOIDS_RESTARTS_DEFAULT, double tolerance = DENSITY_TOLERANCE_DEFAULT) {
  int n = instances.size();
  if(n <= 2) {
    medoids.insert(instances.begin(), instances.end());
    return;
  }

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<std::mt19937::result_type> dist(0, n - 1);

  int best_anchor = -1, best_sample = -1;
  double best_anchor_density = -INFINITY, best_sample_density = -INFINITY;
  vector<int> best_anchor_inst, best_sample_inst;
  double best_score = -INFINITY;

  for(int restart = 0; restart < restarts; restart++) {
    int anchor = instances[dist(gen)];
    int sample = sample_medoid(x, instances, anchor, gen);

    vector<int> anchor_inst, sample_inst;
    auto partition = [&]() {
      anchor_inst.clear();
      sample_inst.clear();

      double total_anchor = 0.0;
      double total_sample = 0.0;
      for(int i = 0; i < n; i++) {
	if(instances[i] != anchor and instances[i] != sample) {
	  double d_anchor = x.distance(anchor, instances[i]);
	  double d_sample = x.distance(sample, instances[i]);
	  total_anchor += d_anchor;
	  total_sample += d_sample;
	  if(d_anchor < d_sample) {
	    anchor_inst.push_back(instances[i]);
	  } else {
	    sample_inst.push_back(instances[i]);
	  }
	}
      }
      return make_pair(total_anchor, total_sample);
    };

    for(int epoch = 0; epoch < epochs; epoch++) {
      auto totals = partition();

      double avg_anchor = totals.first / max(1, (int)anchor_inst.size());
      double avg_sample = totals.second / max(1, (int)sample_inst.size());
      auto new_anchor = select_medoid(anchor, avg_anchor, anchor_inst, x);
      auto new_sample = select_medoid(sample, avg_sample, sample_inst, x);

      if(anchor == new_anchor.first and sample == new_sample.first) break;

      anchor = new_anchor.first;
      sample = new_sample.first;
    }
    // If the loop above hit the epoch cap instead of converging, anchor/
    // sample were just updated to new_anchor/new_sample but anchor_inst/
    // sample_inst still reflect the previous (pre-update) anchor/sample --
    // repartition once more so the medoids and the density() computed
    // below always agree with the same anchor/sample.
    partition();

    double anchor_density = x.density(anchor, anchor_inst);
    double sample_density = x.density(sample, sample_inst);
    double score = anchor_density + sample_density;

    if(score > best_score) {
      best_score = score;
      best_anchor = anchor;
      best_sample = sample;
      best_anchor_density = anchor_density;
      best_sample_density = sample_density;
      best_anchor_inst = anchor_inst;
      best_sample_inst = sample_inst;
    }
  }

  double tolerant_parent_density = parent_density * (1.0 - tolerance);

  if(best_anchor_density > tolerant_parent_density) {
    kmedoids(x, best_anchor_inst, medoids, epochs, best_anchor_density, restarts, tolerance);
  } else {
    medoids.insert(best_anchor);
  }
  if(best_sample_density > tolerant_parent_density) {
    kmedoids(x, best_sample_inst, medoids, epochs, best_sample_density, restarts, tolerance);
  } else {
    medoids.insert(best_sample);
  }
}

