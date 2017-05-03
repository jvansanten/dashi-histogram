
#ifndef HISTOGRAM_H_INCLUDED
#define HISTOGRAM_H_INCLUDED

#include <utility>
#include <iostream>
#include <array>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <cassert>

namespace histogram {

namespace binning {

struct dimension_tag {};

/**
 * @brief A non-equispaced binning scheme
 *
 * In the general case, the proper bin can be found in logarithmic time
 * by binary search
 */
class general : dimension_tag {
public:
	/**
	 * Construct a binning scheme from the given ordered list
	 * of bin edges, inserting under- and overflow bins as necessary
	 */
	general(const std::vector<double> &edges, const std::string &name=std::string())
	    : name_(name)
	{
		if (edges.front() > -std::numeric_limits<double>::infinity())
			edges_.push_back(-std::numeric_limits<double>::infinity());
		std::copy(edges.begin(), edges.end(), std::back_inserter(edges_));
		if (edges.back() < std::numeric_limits<double>::infinity())
			edges_.push_back(std::numeric_limits<double>::infinity());
	}
	
	/** Return the edges of the bins */
	const std::vector<double>& edges() const
	{ return edges_; }
	
	size_t nbins() const { return edges_.size()-1; }
	
	const std::string& name() const { return name_; }
	
	size_t index(double value) const
	{
		long j = (std::distance(edges_.begin(),
		    std::upper_bound(edges_.begin(),
		    edges_.end(), value)));
		assert(j > 0);
		return j-1;
	}
private:
	std::string name_;
	std::vector<double> edges_;
};

namespace detail {

/**
 * @brief Trivial linear mapping
 */
struct identity {
	static inline double map(double v) { return v; }
	static inline double imap(double v) { return v; }
};

/**
 * @brief Bin edges linear in @f$ \log_{10}{x} @f$
 */
struct log10 {
	static inline double map(double v) { return std::pow(10, v); }
	static inline double imap(double v) { return std::log10(v); }
};

/**
 * @brief Bin edges linear in @f$ \cos{\theta} @f$
 */
struct cosine {
	static inline double map(double v) { return std::acos(v); }
	static inline double imap(double v) { return std::cos(v); }
};

/**
 * @brief Bin edges linear in @f$ x^N @f$
 *
 * @tparam N the exponent of the power law
 */
template <int N>
struct power {
	static inline double map(double v) { return std::pow(v, N); }
	static inline double imap(double v) { return std::pow(v, 1./N); }
};

/** @cond */
template <>
struct power<2> {
	static inline double map(double v) { return v*v; }
	static inline double imap(double v) { return std::sqrt(v); }
};
/** @endcond */

}

/**
 * @brief An equispaced binning scheme
 *
 * In this optimal case the bin edges are uniform under some
 * transformation between set limits and the bin index can be
 * found in constant time
 *
 * @tparam Transformation the transformation that makes the bin edges
 *                        equispaced
 */
template <typename Transformation = detail::identity >
class uniform : public dimension_tag {
public:
	uniform(double low, double high, size_t nbins, const std::string &name=std::string())
	    : name_(name), offset_(Transformation::imap(low)),
	    range_(Transformation::imap(high)-Transformation::imap(low)),
	    min_(map(0)), max_(map(1)), nsteps_(nbins+1)
	{
		edges_.reserve(nsteps_+2);
		edges_.push_back(-std::numeric_limits<double>::infinity());
		for (size_t i = 0; i < nsteps_; i++)
			edges_.push_back(map(i/double(nsteps_-1)));
		edges_.push_back(std::numeric_limits<double>::infinity());
	}
	
	/** Return the edges of the bins */
	const std::vector<double>& edges() const
	{ return edges_; }
	
	size_t nbins() const { return nsteps_ + 1; }
	
	size_t index(double value) const
	{
		if (value < min_)
			return 0;
		else if (value >= max_)
			return (edges_.size()-2);
		else {
			return size_t(std::floor((nsteps_-1)*imap(value)))+1;
		}
	}
	
	const std::string& name() const
	{ return name_; }
	
private:
	inline double map(double value) const
	{
		return Transformation::map(range_*value + offset_);
	}
	inline double imap(double value) const
	{
		return (Transformation::imap(value)-offset_)/range_;
	}
	
	std::vector<double> edges_;
	std::string name_;
	double offset_, range_, min_, max_;
	size_t nsteps_;
};

// Convenient typedefs
typedef uniform<> linear;
typedef uniform<detail::log10> log10;
typedef uniform<detail::cosine> cosine;

template <int N>
using power = detail::power<N>;

}

template <class... Ts>
class histogram_impl {
public:
	// recursion endpoints
	size_t index() const { return 0; }
	bool valid() const { return true; }
	size_t stride() const { return 1; }
	size_t size() const { return 1; }

protected:
	// recursion endpoints
	template <typename Value, size_t N>
	void fill_shape(std::array<Value, N> &shape, size_t idx=0) const {}
	template <typename Value, size_t N>
	void fill_edges(std::array<Value, N> &shape, size_t idx=0) const {}
	template <typename Value, size_t N>
	void fill_label(std::array<Value, N> &shape, size_t idx=0) const {}
};

template <class T, class... Ts>
class histogram_impl<T, Ts...> : public histogram_impl<Ts...> {
public:
	static constexpr unsigned Rank  = sizeof...(Ts) + 1;
	histogram_impl(T t, Ts...ts) : histogram_impl<Ts...>(ts...), dimension_(t)
	{}
	
private:
	T dimension_;

protected:
	// typedef std::array<size_t, sizeof...(Ts)+1> coord_type;
	template <typename... Tail>
	size_t index(double v, Tail...tail)
	{
		static_assert(sizeof...(Tail) == sizeof...(Ts), "Number of arguments must match number of dimensions");
		return dimension_.index(v)*stride() + histogram_impl<Ts...>::index(tail...);
	}
	
	template <typename... Tail>
	bool valid(double v, Tail...tail)
	{
		return !std::isnan(v) && histogram_impl<Ts...>::valid(tail...);
	}
	
	size_t stride() const
	{
		return histogram_impl<Ts...>::size();
	}
	
	size_t size() const
	{
		return extent() * histogram_impl<Ts...>::size();
	}
	
	size_t extent() const
	{
		return dimension_.nbins();
	}
	
	// TODO: is there a way to call a member function like this generically?
	template <typename Value, size_t N>
	void fill_shape(std::array<Value, N> &shape, size_t idx=0) const
	{
		shape[idx] = extent();
		histogram_impl<Ts...>::fill_shape(shape, idx+1);
	}
	
	template <typename Value, size_t N>
	void fill_edges(std::array<Value, N> &shape, size_t idx=0) const
	{
		shape[idx] = dimension_.edges();
		histogram_impl<Ts...>::fill_edges(shape, idx+1);
	}
	
	template <typename Value, size_t N>
	void fill_label(std::array<Value, N> &shape, size_t idx=0) const
	{
		shape[idx] = dimension_.name();
		histogram_impl<Ts...>::fill_label(shape, idx+1);
	}
};

namespace detail {

template <typename T, size_t Rank>
struct view {
public:
	view(const T *data, std::array<size_t, Rank> shape) : data_(data), shape_(shape)
	{}
	const T *data_;
	std::array<size_t, Rank> shape_;
};

}

template <class... Dimensions>
class histogram : public histogram_impl<Dimensions...> {
public:
	histogram(Dimensions...dims, const std::string &title=std::string())
	    : histogram_impl<Dimensions...>(dims...), title_(title), n_entries_(0),
	      bincontent_(this->size(), 0.), squaredweights_(this->size(), 0.)
	{}
	
	size_t ndim() const { return sizeof...(Dimensions); }
	const std::string& title() const { return title_; }
	void set_title (const std::string &title) { title_ = title; }
	
	template <typename... Args>
	bool fill(Args...args) {
		return fill_with_weight(1., args...);
	}

	template <typename... Args>
	bool fill_with_weight(double weight, Args...args) {
		static_assert(sizeof...(Args) == sizeof...(Dimensions), "Number of arguments must match number of dimensions");
		
		if (this->valid(args...)) {
			size_t offset = this->index(args...);
			bincontent_.at(offset) += weight;
			squaredweights_.at(offset) += weight*weight;
			n_entries_++;
			return true;
		} else {
			return false;
		}
	}

	std::array<size_t, sizeof...(Dimensions)> shape() const
	{
		std::array<size_t, sizeof...(Dimensions)> shape;
		this->fill_shape(shape);
		return shape;
	}
	
	std::array<std::vector<double>, sizeof...(Dimensions)> binedges() const
	{
		std::array<std::vector<double>, sizeof...(Dimensions)> shape;
		this->fill_edges(shape);
		return std::move(shape);
	}
	
	std::array<std::string, sizeof...(Dimensions)> labels() const
	{
		std::array<std::string, sizeof...(Dimensions)> shape;
		this->fill_label(shape);
		return std::move(shape);
	}
	
	auto bincontent() const
	{ return detail::view<double, sizeof...(Dimensions)>(bincontent_.data(), shape()); }
	
	auto squaredweights() const
	{ return detail::view<double, sizeof...(Dimensions)>(squaredweights_.data(), shape()); }
	
	auto n_entries() const { return n_entries_; }
	
private:
	std::string title_;
	size_t n_entries_;
	std::vector<double> bincontent_, squaredweights_;

};

template<typename... Conds>
  struct and_
  : std::true_type
  { };

template<typename Cond, typename... Conds>
  struct and_<Cond, Conds...>
  : std::conditional<Cond::value, and_<Conds...>, std::false_type>::type
  { };

template <class... Ts>
typename std::enable_if<and_<std::is_base_of<binning::dimension_tag, Ts>... >::value, histogram<Ts...> >::type
create(Ts...ts)
{
	return std::move(histogram<Ts...>(ts...));
}

template <class... Ts>
typename std::enable_if<and_<std::is_base_of<binning::dimension_tag, Ts>... >::value, histogram<Ts...> >::type
create(const std::string &title, Ts...ts)
{
	return std::move(histogram<Ts...>(ts..., title));
}

}

#endif