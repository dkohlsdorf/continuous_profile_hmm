#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "hierarchical_kmedian_dtw.hpp"

namespace py = pybind11;

// DistanceManagement (in the header) stores a raw Dataset* and never takes
// ownership -- fine in C++ where the caller controls the dataset's
// lifetime, but not safe to expose directly to Python, where the dataset
// list could be garbage collected out from under it. This wrapper owns a
// copy of the dataset alongside the DistanceManagement that points into
// it, so the pointer stays valid for the wrapper's whole lifetime.
class DistanceManager {
public:
  DistanceManager(Dataset dataset, int warping_band, bool normalize)
      : dataset(std::move(dataset)), impl(&this->dataset, warping_band, normalize) {}

  double distance(int i, int j) {
    return impl.distance(i, j);
  }

  int size() {
    return impl.size();
  }

  // Divisive hierarchical k-medoids/k-median clustering under DTW distance
  // (see kmedoids() in hierarchical_kmedian_dtw.hpp). Recursively bisects
  // `instances`, stopping a branch once splitting it further would not
  // produce a denser (tighter) pair of children than the branch itself,
  // then returns the leaf medoids -- an empty `instances` clusters every
  // sequence in the dataset. restarts/tolerance default to the same
  // values hierarchical_kmedian_dtw.hpp itself defaults to.
  std::vector<int> kmedoids(std::vector<int> instances, int epochs, int restarts, double tolerance) {
    if (instances.empty()) {
      instances.resize(impl.size());
      for (int i = 0; i < impl.size(); i++) instances[i] = i;
    }
    std::set<int> medoids;
    ::kmedoids(impl, instances, medoids, epochs, 0.0, restarts, tolerance);
    return std::vector<int>(medoids.begin(), medoids.end());
  }

private:
  Dataset dataset;
  DistanceManagement impl;
};


PYBIND11_MODULE(hierarchical_kmedian_dtw, m) {
  m.doc() = "Divisive hierarchical k-medoids clustering under DTW distance";

  m.def("dtw", [](Mat x, Mat y, int warping_band, bool normalize) { return dtw(x, y, warping_band, normalize); },
        py::arg("x"), py::arg("y"), py::arg("warping_band"), py::arg("normalize") = true,
        "Sakoe-Chiba banded DTW distance between two sequences. normalize=true "
        "divides by the warp-path length, false returns the summed cost.");

  py::class_<DistanceManager>(m, "DistanceManager")
    .def(py::init<Dataset, int, bool>(), py::arg("dataset"), py::arg("warping_band"),
         py::arg("normalize") = true,
         "dataset: list of sequences (each a list of frames/vectors). "
         "Pairwise DTW distances are computed lazily and cached. "
         "normalize: divide each distance by its warp-path length (see dtw()).")
    .def("distance", &DistanceManager::distance, py::arg("i"), py::arg("j"),
         "Cached DTW distance between dataset[i] and dataset[j].")
    .def("size", &DistanceManager::size)
    .def("kmedoids", &DistanceManager::kmedoids,
         py::arg("instances") = std::vector<int>{}, py::arg("epochs") = 20,
         py::arg("restarts") = KMEDOIDS_RESTARTS_DEFAULT, py::arg("tolerance") = DENSITY_TOLERANCE_DEFAULT,
         "Divisively cluster `instances` (dataset indices; empty = all) by "
         "DTW distance, splitting in two by k-medoids each level, recursing "
         "into a child only while it is denser (tighter around its medoid) "
         "than its parent -- no external distance/count threshold needed, "
         "the split count is entirely data-driven. `restarts` random "
         "(anchor, sample) attempts are tried per split and the densest is "
         "kept; `tolerance` is the fractional slack allowed below the "
         "parent's density before a branch stops (0.05 = a child up to 5% "
         "less dense than its parent still keeps splitting). Returns the "
         "resulting leaf medoids as dataset indices.");
}
