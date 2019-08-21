#include "BodyModel.h"

void BodyModel::load(const std::string &_folder)
{
	std::ifstream f;

	//--------------------------------------------------------------------------------
	// read points coordinates
	f.open(_folder + "/" + verticesFileName);
	if (!f.is_open())
	{
		std::cout << "Cannot open:" << _folder + "/" + verticesFileName << std::endl;
		exit(-1);
	}

	m_vertex.resize(m_verticeNum);
	m_vertex_new.resize(m_verticeNum);

	for (int i = 0; i < m_verticeNum; i++)
	{
		f >> m_vertex[i].x() >> m_vertex[i].y() >> m_vertex[i].z();
	}
	f.close();

	//--------------------------------------------------------------------------------
	// read faces
	f.open(_folder + "/" + facesFileName);
	if (!f.is_open())
	{
		std::cout << "Cannot open:" << _folder + "/" + verticesFileName << std::endl;
		exit(-1);
	}
	m_faces.resize(m_faceNum);
	for (int i = 0; i < m_faceNum; i++)
	{
		double v0, v1, v2;
		f >> v0 >> v1 >> v2;
		m_faces[i].x() = static_cast<int>(v0);
		m_faces[i].y() = static_cast<int>(v1);
		m_faces[i].z() = static_cast<int>(v2);
	}
	f.close();

	m_normals.resize(m_verticeNum);
	m_normals_new.resize(m_verticeNum);
	calculateNormals();

	//--------------------------------------------------------------------------------
	// read kin-tree
	f.open(_folder + "/" + kintreeFileName);
	if (!f.is_open())
	{
		std::cout << "Cannot open:" << _folder + "/" + verticesFileName << std::endl;
		exit(-1);
	}
	m_kintree.resize(m_jointNum);
	for (int i = 0; i < m_jointNum; i++)
	{
		double tmp;
		f >> tmp;
		m_kintree[i].x() = static_cast<int>(tmp);
	}
	for (int i = 0; i < m_jointNum; i++)
	{
		double tmp;
		f >> tmp;
		m_kintree[i].y() = static_cast<int>(tmp);
	}
	f.close();

	//--------------------------------------------------------------------------------
	// read joints
	f.open(_folder + "/" + jointsFileName);
	if (!f.is_open())
	{
		std::cout << "Cannot open:" << _folder + "/" + verticesFileName << std::endl;
		exit(-1);
	}
	m_joints.resize(m_jointNum);
	for (int i = 0; i < m_jointNum; i++)
	{
		f >> m_joints[i].x() >> m_joints[i].y() >> m_joints[i].z();
	}
	f.close();

	//--------------------------------------------------------------------------------
	// read joint regressor
	f.open(_folder + "/" + jregressorFileName);
	if (!f.is_open())
	{
		std::cout << "Cannot open:" << _folder + "/" + jregressorFileName << std::endl;
		exit(-1);
	}
	std::vector<Eigen::Triplet<float>> jregressorList;
	for (int i = 0; i < m_boneNum + 1; i++)
	{
		for (int j = 0; j < m_verticeNum; j++)
		{
			float elem;
			f >> elem;
			if (elem > 1e-6)
				jregressorList.push_back(Eigen::Triplet<float>(i, j, elem));
		}
	}
	f.close();
	m_jregressor.resize(m_boneNum + 1, m_verticeNum);
	m_jregressor.setFromTriplets(jregressorList.begin(), jregressorList.end());

	//--------------------------------------------------------------------------------
	// read shape blending parameters
	f.open(_folder + "/" + shapeParamFileName);
	if (!f.is_open())
	{
		std::cout << "Cannot open:" << _folder + "/" + shapeParamFileName << std::endl;
		exit(-1);
	}
	std::vector<float> shapeParamsHost(m_verticeNum * 30);
	m_shapeParams.resize(m_verticeNum);
	for (int i = 0; i < m_verticeNum; i++)
	{
		for (int j = 0; j < 3; j++)
			for (int k = 0; k < m_shapeParamNum; k++)
			{
				f >> m_shapeParams[i].coeffRef(j, k);
				shapeParamsHost[(j * 10 + k)*m_verticeNum + i] = m_shapeParams[i].coeffRef(j, k);
			}
	}
}

void BodyModel::changeShapeAB(
	const Eigen::VectorXf &_shapeCoeffs,
	const std::string &_shapeCode,
	std::vector<float> &_ratios,
	std::vector<std::pair<std::string, float>> &_weird_ratios)
{
	//----------------------
	// calc new shape vertex
	for (int vidx = 0; vidx < m_verticeNum; vidx++)
	{
		VerticeShapeMatrix m = m_shapeParams[vidx];
		Eigen::Vector3f v_shift = m * _shapeCoeffs;
		m_vertex[vidx] += v_shift;
	}
	calculateNormals();

	//-------------------------
	// calc joints of new shape
	Eigen::Matrix<float, BodyModelParam::verticeNum, 3> vertexMatrix;
	for (int i = 0; i < m_verticeNum; i++)
	{
		vertexMatrix.coeffRef(i, 0) = m_vertex[i][0];
		vertexMatrix.coeffRef(i, 1) = m_vertex[i][1];
		vertexMatrix.coeffRef(i, 2) = m_vertex[i][2];
	}
	Eigen::Matrix<float, BodyModelParam::jointNum + 1, 3> jointCoord = m_jregressor * vertexMatrix;
	for (int i = 0; i < m_jointNum; i++)
	{
		m_joints[i] = jointCoord.row(i).transpose();
	}

	//-------------------------------------
	// calc arm skeleton ratio of new shape
	Eigen::Vector3f shoulder = m_joints[16];
	Eigen::Vector3f elbow = m_joints[18];
	Eigen::Vector3f wrist = m_joints[20];

	float upper_length = (elbow - shoulder).norm();
	float lower_length = (wrist - elbow).norm();
	float ratio = upper_length / lower_length;

	_ratios.push_back(ratio);

	if (ratio >= 1.0f || ratio <= 0.90f)
	{
		_weird_ratios.push_back(std::pair<std::string, float>(_shapeCode, ratio));
		//if (_shapeCode == "1110110111" || _shapeCode == "1112112111")
		//{
		//	int i_ratio = (int)(ratio * 1e6);
		//	std::string mesh_path = "./results/smpl_" + _shapeCode + std::string("_") + std::to_string(i_ratio) + "_.obj";
		//	saveAsObj(mesh_path);
		//}
	}

	// reset vertex to mean shape
	for (int vidx = 0; vidx < m_verticeNum; vidx++)
	{
		VerticeShapeMatrix m = m_shapeParams[vidx];
		Eigen::Vector3f v_shift = m * _shapeCoeffs;
		m_vertex[vidx] -= v_shift;
	}
}

void BodyModel::changeShapeAB(
	const Eigen::VectorXf &_shapeCoeffs)
{
	//----------------------
	// calc new shape vertex
	for (int vidx = 0; vidx < m_verticeNum; vidx++)
	{
		VerticeShapeMatrix m = m_shapeParams[vidx];
		Eigen::Vector3f v_shift = m * _shapeCoeffs;
		m_vertex_new[vidx] = m_vertex[vidx] +  v_shift;
	}
	calculateNormalsNew();
}


void BodyModel::saveAsObj(const std::string &_path)
{
	//--------------------------------------------------------------------------------
	// save mesh (with faces)
	std::ofstream f(_path);
	for (int i = 0; i < m_verticeNum; i++)
	{
		f << "v "
			<< m_vertex[i].x() << " "
			<< m_vertex[i].y() << " "
			<< m_vertex[i].z() << std::endl;
	}
	for (int i = 0; i < m_faceNum; i++)
	{
		// Note that faces are 1-based, not 0-based in obj files
		f << "f "
			<< m_faces[i].x() + 1 << " "
			<< m_faces[i].y() + 1 << " "
			<< m_faces[i].z() + 1 << std::endl;
	}
	f.close();
}

void BodyModel::calculateNormals()
{
	//---------------------------------------------------------------------
	// normals
	for (int verticeIdx = 0; verticeIdx < m_vertex.size(); verticeIdx++)
	{
		m_normals[verticeIdx] = Eigen::Vector3f::Zero();
	}

	for (int faceIdx = 0; faceIdx < m_faces.size(); faceIdx++)
	{
		Eigen::Vector3i idxes = m_faces[faceIdx];
		Eigen::Vector3f pt1 = m_vertex[idxes.x()];
		Eigen::Vector3f pt2 = m_vertex[idxes.y()];
		Eigen::Vector3f pt3 = m_vertex[idxes.z()];

		Eigen::Vector3f normal = (pt1 - pt2).cross(pt2 - pt3);
		normal.normalize();
		//if (normal.z() < 0) normal = -normal;

		m_normals[idxes.x()] += normal;
		m_normals[idxes.y()] += normal;
		m_normals[idxes.z()] += normal;
	}

	for (int verticeIdx = 0; verticeIdx < m_vertex.size(); verticeIdx++)
	{
		Eigen::Vector3f n = m_normals[verticeIdx];
		float length = sqrt(n.x()*n.x() + n.y()*n.y() + n.z()*n.z());
		m_normals[verticeIdx] *= 1.0f / length;
	}
}

void BodyModel::calculateNormalsNew()
{
	//---------------------------------------------------------------------
	// normals
	for (int verticeIdx = 0; verticeIdx < m_vertex_new.size(); verticeIdx++)
	{
		m_normals_new[verticeIdx] = Eigen::Vector3f::Zero();
	}

	for (int faceIdx = 0; faceIdx < m_faces.size(); faceIdx++)
	{
		Eigen::Vector3i idxes = m_faces[faceIdx];
		Eigen::Vector3f pt1 = m_vertex_new[idxes.x()];
		Eigen::Vector3f pt2 = m_vertex_new[idxes.y()];
		Eigen::Vector3f pt3 = m_vertex_new[idxes.z()];

		Eigen::Vector3f normal = (pt1 - pt2).cross(pt2 - pt3);
		normal.normalize();
		//if (normal.z() < 0) normal = -normal;

		m_normals_new[idxes.x()] += normal;
		m_normals_new[idxes.y()] += normal;
		m_normals_new[idxes.z()] += normal;
	}

	for (int verticeIdx = 0; verticeIdx < m_vertex.size(); verticeIdx++)
	{
		Eigen::Vector3f n = m_normals_new[verticeIdx];
		float length = sqrt(n.x()*n.x() + n.y()*n.y() + n.z()*n.z());
		m_normals_new[verticeIdx] *= 1.0f / length;
	}
}