

#include "histogram.h"
#include "histogram_storage.h"

template <template <typename, typename...> class ContainerType,
          typename ValueType, typename... Args>
void print_container(const ContainerType<ValueType, Args...>& c) {
  for (const auto& v : c) {
    std::cout << v << ' ';
  }
  std::cout << '\n';
}

template <typename T, size_t N>
std::ostream& operator<<(std::ostream &s, const std::array<T,N> &a)
{
	s << "[ ";
	for (auto &v: a)
		s << v << ", ";
	s << ']';
	return s;
}

template <typename T>
std::ostream& operator<<(std::ostream &s, const std::vector<T> &a)
{
	s << "[ ";
	for (auto &v: a)
		s << v << ", ";
	s << ']';
	return s;
}

int main (int argc, char const *argv[])
{
	auto dim = histogram::binning::linear(0, 10, 11, "dimension");
	auto dim1 = histogram::binning::general({0, 1, 2}, "general");
	auto f = histogram::create("hola cabrones", dim, dim1);
	
	f.fill(1., 1.);
	
	std::cout << "shape: " << f.shape() << std::endl;
	std::cout << "labels: " << f.labels() << std::endl;
	std::cout << "binedges: " << f.binedges() << std::endl;
	
	std::cout << "saving histogram to foo.hdf5" << std::endl;
	histogram::save(f, "foo.hdf5", "/", "foo", true);
	
	return 0;
}