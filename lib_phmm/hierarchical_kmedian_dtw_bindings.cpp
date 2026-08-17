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
  DistanceManager(Dataset dataset, int warping_band)
      : dataset(std::move(dataset)), impl(&this->dataset, warping_band) {}

  double distance(int i, int j) {
    return impl.distance(i, j);
  }

  int size() {
    return impl.size();
  }

  // Divisive hierarchical k-medoids/k-median clustering under DTW distance
  // (see kmedoids() in hierarchical_kmedian_dtw.hpp). Recursively bisects
  // `instances` until each leaf's average intra-cluster distance drops
  // below `threshold`, then returns the leaf medoids -- an empty
  // `instances` clusters every sequence in the dataset.
  std::vector<int> kmedoids(std::vector<int> instances, int epochs, double threshold) {
    if (instances.empty()) {
      instances.resize(impl.size());
      for (int i = 0; i < impl.size(); i++) instances[i] = i;
    }
    std::set<int> medoids;
    ::kmedoids(impl, instances, medoids, epochs, threshold);
    return std::vector<int>(medoids.begin(), medoids.end());
  }

private:
  Dataset dataset;
  DistanceManagement impl;
};


PYBIND11_MODULE(hierarchical_kmedian_dtw, m) {
  m.doc() = "Divisive hierarchical k-medoids clustering under DTW distance";

  m.def("dtw", [](Mat x, Mat y, int warping_band) { return dtw(x, y, warping_band); },
        py::arg("x"), py::arg("y"), py::arg("warping_band"),
        "Sakoe-Chiba banded DTW distance between two sequences.");

  py::class_<DistanceManager>(m, "DistanceManager")
    .def(py::init<Dataset, int>(), py::arg("dataset"), py::arg("warping_band"),
         "dataset: list of sequences (each a list of frames/vectors). "
         "Pairwise DTW distances are computed lazily and cached.")
    .def("distance", &DistanceManager::distance, py::arg("i"), py::arg("j"),
         "Cached DTW distance between dataset[i] and dataset[j].")
    .def("size", &DistanceManager::size)
    .def("kmedoids", &DistanceManager::kmedoids,
         py::arg("instances") = std::vector<int>{}, py::arg("epochs") = 20, py::arg("threshold") = 0.0,
         "Divisively cluster `instances` (dataset indices; empty = all) by "
         "DTW distance, splitting in two by k-medoids each level until a "
         "branch's average intra-cluster distance to its medoid is below "
         "`threshold`. Returns the resulting leaf medoids as dataset indices.");
}
