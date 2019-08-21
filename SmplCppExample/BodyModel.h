#pragma once
#include <Eigen/Eigen>
#include <Eigen/Sparse>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>

// Parameters of body model
struct BodyModelParam
{
	const static int jointNum = 23;
	const static int verticeNum = 6890;
	const static int faceNum = 13776;
	const static int shapeParamNum = 10;
	const static int gmmComponentNum = 8;
};

typedef Eigen::Matrix<float, 3, BodyModelParam::shapeParamNum> VerticeShapeMatrix;

class BodyModel
{
public:
	BodyModel() {}
	void load(const std::string &_folder);
	void saveAsObj(const std::string &_path);
	void changeShapeAB(
		const Eigen::VectorXf &_shapeCoeffs,
		const std::string &_shapeCode,
		std::vector<float> &_ratios,
		std::vector<std::pair<std::string, float>> &_weird_ratios);
	void changeShapeAB(const Eigen::VectorXf &_shapeCoeffs);

	std::vector<Eigen::Vector3f> m_vertex;
	std::vector<Eigen::Vector3f> m_normals;
	std::vector<Eigen::Vector3i> m_faces;

	std::vector<Eigen::Vector3f> m_vertex_new;
	std::vector<Eigen::Vector3f> m_normals_new;
	std::vector<Eigen::Vector3i> m_faces_new;

	void calculateNormals();
	void calculateNormalsNew();

protected:
private:

	//--------------------------------------------------------------------------------
	// file name
	const char *facesFileName = "faces.txt";
	const char *jointsFileName = "jointsFile.txt";
	const char *keypointsFileName = "keypoint.txt";
	const char *kintreeFileName = "kintreeFile.txt";
	const char *poseParamFileName = "poseParams.txt";
	const char *shapeParamFileName = "shapeParams.txt";
	const char *verticesFileName = "vertices.txt";
	const char *weightsFileName = "weightsFile.txt";
	const char *jregressorFileName = "JRegressor.txt";
	const char *freedonLambdaFileName = "freedomLambda.txt";
	const char *eliminateFlagFilename = "eliminationFlagFile.txt";

	const int m_verticeNum = 6890;
	const int m_faceNum = 13776;
	const int m_boneNum = 23;
	const int m_jointNum = 24;
	const int m_shapeParamNum = 10;

	std::vector<VerticeShapeMatrix> m_shapeParams;  // shape blend parameters
	Eigen::SparseMatrix<float> m_jregressor;        // joint regression matrix
	std::vector<Eigen::Vector2i> m_kintree;         // kin-tree of joints
	std::vector<Eigen::Vector3f> m_joints;          // joints of body
};