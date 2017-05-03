
#ifndef HISTOGRAM_HISTSTORAGE_H_INCLUDED
#define HISTOGRAM_HISTSTORAGE_H_INCLUDED

#include "simple_hdf5.hpp"
#include <sstream>
#include <algorithm>

namespace histogram {

namespace detail {

// storage adapters for detail::view
template <typename T, size_t N>
const void*
get_data(const view<T,N> &d) { return d.data_; }

template <typename T, size_t N>
std::vector<hsize_t>
get_shape(const view<T,N> &d)
{
	std::vector<hsize_t> shape(d.shape_.size());
	std::copy(d.shape_.begin(), d.shape_.end(), shape.begin());
	return std::move(shape);
}

template <typename T, size_t N>
hdf5::Datatype
get_datatype(const view<T,N> &d) { return hdf5::get_datatype(T()); }

}

// cribbed from: http://stackoverflow.com/a/11329249
namespace detail {

template<typename Iterable, bool rvalue>
class enumerate_object
{
private:
    typedef typename std::conditional<rvalue, const Iterable&, Iterable>::type IterableType;
    IterableType _iter;
    std::size_t _size;
    decltype(std::begin(_iter)) _begin;
    const decltype(std::end(_iter)) _end;

public:
    enumerate_object(const Iterable& iter):
        _iter(iter),
        _size(0),
        _begin(std::begin(_iter)),
        _end(std::end(_iter))
    {}

    enumerate_object(const Iterable&& iter):
        _iter(std::move(iter)),
        _size(0),
        _begin(std::begin(_iter)),
        _end(std::end(_iter))
    {}

    const enumerate_object& begin() const { return *this; }
    const enumerate_object& end()   const { return *this; }

    bool operator!=(const enumerate_object&) const
    {
        return _begin != _end;
    }

    void operator++()
    {
        ++_begin;
        ++_size;
    }

    auto operator*() const
        -> std::pair<std::size_t, decltype(*_begin)>
    {
        return { _size, *_begin };
    }
};

}

// lvalue; take a reference
template<typename Iterable>
detail::enumerate_object<Iterable, true> enumerate(const Iterable& iter)
{
    return { iter };
}

// lvalue; adopt held object
template<typename Iterable>
detail::enumerate_object<Iterable, false> enumerate(const Iterable&& iter)
{
    return { iter };
}

template <typename T>
void save(const T& hist, hdf5::File file, const std::string &where, const std::string &name, bool overwrite=false)
{
	using namespace hdf5;
	
	Group group = file.create_group(where, name, true);
	auto attr = group.attrs();
	
	attr["ndim"] = hist.ndim();
	attr["nentries"] = hist.n_entries();
	attr["title"] = hist.title();
	
	file.create_carray(group, "_h_bincontent", hist.bincontent());
	file.create_carray(group, "_h_squaredweights", hist.squaredweights());
	for (const auto &pair : enumerate(hist.binedges())) {
		std::ostringstream ss;
		ss << "_h_binedges_" << pair.first;
		file.create_carray(group, ss.str(), pair.second);
	}
	for (const auto &pair : enumerate(hist.labels())) {
		std::ostringstream ss;
		ss << "label_" << pair.first;
		attr[ss.str()] = pair.second;
	}
}

template <typename T>
void save(const T& hist, const std::string &fname, const std::string &where, const std::string &name, bool overwrite=false)
{
	save(hist, hdf5::open_file(fname, hdf5::File::append), where, name, overwrite);
}

}

#endif // HISTOGRAM_HISTSTORAGE_H_INCLUDED
