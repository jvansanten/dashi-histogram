
#ifndef SIMPLE_HDF5_H_INCLUDED
#define SIMPLE_HDF5_H_INCLUDED

#include <H5Dpublic.h>
#include <H5Epublic.h>
#include <H5Fpublic.h>
#include <H5Gpublic.h>
#include <H5Tpublic.h>
#include <H5Spublic.h>
#include <H5Apublic.h>
#include <H5Ppublic.h>

#include <sstream>

template <typename T>
T clamp(const T& v, const T& a, const T& b)
{
  return std::max(a, std::min(v, b));
}

namespace hdf5 {

/// @brief Temporarily disable HDF5 error printing
class mute_errors {
public:
	mute_errors() {
		H5Eget_auto2(H5E_DEFAULT, &old_func, &old_client_data);
		H5Eset_auto2(H5E_DEFAULT, NULL, NULL);
	}
	~mute_errors() {
		H5Eset_auto2(H5E_DEFAULT, old_func, old_client_data);
	}
private:
	herr_t (*old_func)(hid_t,void*);
	void *old_client_data;
};

/// @brief A reference-counted handle
class handle {
public:
	typedef int (*close_func)(hid_t);
	/// @brief Initialize empty handle
	handle() : id_(-1), close_(NULL) {}
	/// @brief Initialize handle
	/// @param[in] id Handle to adopt
	/// @param[in] close Function to release handle
	/// The close function will be called to release the object when the
	/// reference count drops to 0
	handle(hid_t id, close_func close) : id_(id), close_(close)
	{
		if (id_ < 0) {
			close_ = NULL;
			throw std::runtime_error("A library call failed!");
		}
	}
	/// @brief Copy handle, increasing reference count
	handle(const handle &other) : id_(other.id_), close_(other.close_)
	{
		if (id_ > 0) {
			incref();
		}
	}
	/// @brief Move handle, keeping reference count constant
	handle(handle &&other) : id_(other.id_), close_(other.close_)
	{
		other.id_ = -1;
		other.close_ = NULL;
	}
	/// @brief Destroy handle
	/// If this is the last remaining reference to the held object, it will be
	/// freed.
	~handle()
	{
		if (refcount() > 1) {
			decref();
		} else if (close_ != NULL && id_ > 0) {
			(*close_)(id_);
		}
	}
	operator hid_t() const { return id_; }
	/// @brief Is this handle still valid?
	explicit operator bool() const { return id_ > 0; }
	
	/// @brief Get the path name 
	std::string name() const
	{
		mute_errors muzzle;
		std::string name;
		ssize_t size = H5Iget_name(*this, NULL, 0);
		if (size > 0) {
			name.resize(size+1);
			H5Iget_name(*this, &name[0], size+1);
		}
		return name;
	}
private:
	hid_t id_;
	close_func close_;
	
	int refcount() { return id_ > 0 ? H5Iget_ref(id_) : 0; }
	void incref() { H5Iinc_ref(id_); }
	void decref() { H5Idec_ref(id_); }
};

class Dataspace : public handle {
public:
	Dataspace(std::vector<hsize_t> &&dims) :
	    handle(create(dims), H5Sclose)
	{}
	bool operator==(const Dataspace &other) const {
		if (hid_t(*this) == hid_t(other))
			return true;
		std::vector<hsize_t> extent(this->get_extent()), other_extent(other.get_extent());
		return (extent.size() == other_extent.size()) &&
		    std::equal(extent.begin(),extent.end(),other_extent.begin());
	}
private:
	std::vector<hsize_t> get_extent() const
	{
		std::vector<hsize_t> dims;
		int size = H5Sget_simple_extent_ndims(*this);
		if (size > 0) {
			dims.resize(size);
			H5Sget_simple_extent_dims(*this, dims.data(), NULL);
		}
		return dims;
	}
	
	static hid_t create(std::vector<hsize_t> &dims)
	{
		if (dims.size() == 0) {
			return H5Screate(H5S_SCALAR);
		} else {
			return H5Screate_simple(dims.size(), dims.data(), NULL);
		}
	}
};

class Datatype : public handle {
public:
	using handle::handle;
};

/// @brief Return the corresponding HDF5 data type
/// Specialize this for custom types
template <typename T>
Datatype
get_datatype(const T&);

template <>
Datatype
get_datatype(const double&) { return Datatype(H5T_NATIVE_DOUBLE, NULL); }

template <>
Datatype
get_datatype(const unsigned long&) { return Datatype(H5T_NATIVE_ULONG, NULL); }

template <>
Datatype
get_datatype(const std::string &v)
{
	hid_t dt = H5Tcopy(H5T_C_S1);
	H5Tset_size(dt, v.size()+1);
	H5Tset_strpad(dt, H5T_STR_NULLTERM);
	return Datatype(dt, H5Tclose);
}

template<>
Datatype
get_datatype(const std::vector<double> &v)
{ return get_datatype(double()); }

/// @brief Return the dimensions of the given object
template <typename T>
typename std::enable_if<!std::is_pod<T>::value, std::vector<hsize_t> >::type
get_shape(const T&);

template <typename T>
typename std::enable_if<std::is_pod<T>::value, std::vector<hsize_t> >::type
get_shape(const T &v)
{ return std::vector<hsize_t>(); }

template <>
std::vector<hsize_t>
get_shape(const std::string &v)
{ return std::vector<hsize_t>(); }

template <>
std::vector<hsize_t>
get_shape(const std::vector<double> &v)
{ return std::vector<hsize_t> {v.size()}; }


/// @brief Calculate an optimal chunk shape of the given size
template <typename T>
std::vector<hsize_t>
get_chunk_shape(const T& data, hsize_t max_chunk_size=size_t(2)<<15)
{
	std::vector<hsize_t> shape(get_shape(data));
	std::vector<hsize_t> chunk(shape.size());
	hsize_t size = H5Tget_size(get_datatype(data));
	for(hsize_t d : shape)
		size *= d;
	hsize_t chunk_size(std::min(max_chunk_size, size));
	hsize_t cum_size = 1;
	for (int i=shape.size()-1; i >= 0; i--) {
		chunk[i] = clamp(chunk_size/cum_size, hsize_t(1), shape[i]);
		cum_size *= shape[i];
	}
	
	return chunk;
}

/// @brief Return a pointer to the beginning of the object's continguous backing store
template <typename T>
const typename std::enable_if<!std::is_pod<T>::value, void>::type*
get_data(const T&);

template <typename T>
const typename std::enable_if<std::is_pod<T>::value, void>::type*
get_data(const T &v)
{ return &v; }

template <>
const void*
get_data(const std::string &v)
{ return v.c_str(); }

template<>
const void*
get_data(const std::vector<double> &v)
{ return v.data(); }

/// @brief A named path
class Node : public handle {
public:
	using handle::handle;
	
	class AttributeSet;
	class AttributeProxy {
	private:
		AttributeProxy(handle &node, const std::string &name) : name_(name), node_(node)
		{}
		friend class AttributeSet;
	
	public:
		template <typename T>
		T operator=(const T &value)
		{
			if (H5Aexists(node_, name_.c_str()) > 0) {
				H5Adelete(node_, name_.c_str());
			}
			Dataspace dspace(get_shape(value));
			Datatype dtype(get_datatype(value));
			
			hid_t attr_id = H5Acreate2(node_, name_.c_str(), dtype, dspace, H5P_DEFAULT, H5P_DEFAULT);
			H5Awrite(attr_id, dtype, get_data(value));
			
			return value;
		}
	private:
		std::string name_;
		handle node_;
	};
	
	class AttributeSet {
	private:
		AttributeSet(handle parent) : parent_(parent)
		{}
		friend class Node;
	public:
		AttributeProxy operator[](const std::string &name)
		{
			return AttributeProxy(parent_, name);
		}
	private:
		handle parent_;
	};
	
	AttributeSet attrs() const { return AttributeSet(*this); }
private:
	handle parent_;
};

/// @brief A group
class Group : public Node {
public:
	using Node::Node;
	hsize_t num_children() 
	{
		hsize_t n;
		H5Gget_num_objs(*this, &n);
		return n;
	}
};

/// @brief A property list specification
class PropertyListClass : public handle {
public:
	using handle::handle;
};

/// @brief A generic property list
class PropertyList : public handle {
public:
	PropertyList() : handle(H5P_DEFAULT, NULL)
	{}
protected:
	PropertyList(hid_t type) : handle(H5Pcreate(type), H5Pclose)
	{}
};

/// @brief Settings for dataset creation
class DatasetCreationProperties : public PropertyList {
public:
	DatasetCreationProperties() : PropertyList(H5P_DATASET_CREATE) {}
	void set_chunk(const std::vector<hsize_t> &chunk) { H5Pset_chunk(*this, chunk.size(), chunk.data()); }
	void set_deflate(unsigned int complevel) { H5Pset_deflate(*this, complevel); }
	void set_shuffle() { H5Pset_shuffle(*this); }
};

/// @brief A dataset
class Dataset : public Node {
public:
	/// @brief Create a dataset
	Dataset(Group group, const std::string &name, Datatype dtype, Dataspace dspace,
	    PropertyList link=PropertyList(), PropertyList creation=PropertyList(),
	    PropertyList access=PropertyList()) :
	    Node(H5Dcreate2(group, name.c_str(), dtype, dspace, link, creation, access), H5Dclose),
	    parent_(group)
	{
		// TODO: check return value
	}
	/// @brief Write data to dataset
	template <typename T>
	void write(const T& data)
	{
		Datatype dtype(get_datatype(data));
		Dataspace dspace(get_shape(data));
		// TODO: check return value
		H5Dwrite(*this, dtype, dspace, H5S_ALL, H5P_DEFAULT, get_data(data));
	}
private:
	Group parent_;
};

class File : public handle {
public:
	using handle::handle;
	enum access { read, write, append };
public:
	Group create_group(const std::string &where, const std::string &name, bool overwrite=false)
	{
		Group root = File::get_group(where);
		if (overwrite) {
			mute_errors muzzle;
			if (H5Gunlink(root, name.c_str()) < 0)
				H5Eclear2(H5E_DEFAULT);
		}
		return Group(H5Gcreate(root, name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose);
	}
	
	template <typename T>
	Dataset create_carray(const std::string &where, const std::string &name, const T& object, bool overwrite=false)
	{
		Group group(H5Gopen(*this, where.c_str(), H5P_DEFAULT), H5Gclose);
		return create_carray(group, name, object, overwrite);
	}
	/// @brief Create a chunked, compressed array
	/// @param[in] where parent group
	/// @param[in] name name of array
	/// @param[in] object object to store. This determines the type and shape of the resulting array
	/// @param[in] overwrite overwrite the existing dataset
	template <typename T>
	Dataset create_carray(Group where, const std::string &name, const T& object, bool overwrite=false)
	{
		
		Dataspace dspace(get_shape(object));
		Datatype dtype(get_datatype(object));
		
		DatasetCreationProperties plist;
		plist.set_chunk(get_chunk_shape(object));
		plist.set_shuffle();
		plist.set_deflate(6);
		
		if (overwrite) {
			mute_errors muzzle;
			if (H5Gunlink(where, name.c_str()) < 0)
				H5Eclear2(H5E_DEFAULT);
		}
		Dataset dataset(where, name, dtype, dspace, PropertyList(), plist, PropertyList());
		dataset.write(object);
		
		return dataset;
	}
	
private:
	Group get_group(const std::string &where)
	{
		mute_errors muzzle;
		hid_t gid = H5Gopen(*this, where.c_str(), H5P_DEFAULT);
		if (gid < 0) {
			// Node does not exist. Walk the requested path, creating parents
			// as necessary.
			hid_t parent = *this;
			std::istringstream iss(where);
			std::string chunk;
			while (std::getline(iss, chunk, '/')) {
				if (chunk.size() == 0)
					continue;
				gid = H5Gopen(parent, chunk.c_str(), H5P_DEFAULT);
				if (gid < 0) {
					gid = H5Gcreate(parent, chunk.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
				}
				H5Gclose(parent);
				parent = gid;
			}
			H5Eclear2(H5E_DEFAULT);
		}
		
		return Group(gid, H5Gclose);
	}
};

/// @brief Open a file for reading or writing
File open_file(const std::string &fname, File::access mode=File::read)
{
	mute_errors muzzle;
	hid_t id;
	if (mode == File::read) {
		id = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
	} else if (mode == File::write) {
		id = H5Fcreate(fname.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
	} else {
		id = H5Fopen(fname.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
		if (id < 0) {
			H5Eclear2(H5E_DEFAULT);
			id = H5Fcreate(fname.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
		}
	}
	if (id < 0) {
		throw std::runtime_error("Couldn't create file");
	}
	return File(id, H5Fclose);
}

}

#endif // SIMPLE_HDF5_H_INCLUDED
