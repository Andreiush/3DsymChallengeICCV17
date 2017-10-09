#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <random>
#include <list>
#include <ctime>
#include <queue>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include  <eigen3/Eigen/Dense>
#include  <eigen3/Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/flann/flann.hpp"
#include "tinyply.h"

using namespace tinyply;
using namespace std;
using namespace Eigen;


namespace
{
inline bool inRange(Vector3d p, int w, int h, int d)
{
    return(p[0]>=0 && p[0]<w && p[1]>=0 && p[1]<h && p[2]>=0 && p[2]<d);
}
inline bool inRange(Vector3f p, Vector3f boxPos, double maxDist)
{
    int w = boxPos[0]+maxDist;
    int h = boxPos[1]+maxDist;
    int d = boxPos[2]+maxDist;
    return(p[0]>=boxPos[0] && p[0]<w && p[1]>=boxPos[1] && p[1]<h && p[2]>=boxPos[2] && p[2]<d);
}

void writePlane(string name, vector<Vector3f> points, bool append=false)
{
    using namespace std;
    std::ofstream os;

    os.open(name.c_str(), append ? ios::app : ios::out);
    os.precision(15);

    if (!append) os <<"#VRML V2.0 utf8" << endl;
    os <<"Shape {"<<endl;
    os <<         " appearance Appearance {"<< endl;
    os <<            "material Material {"<< endl;
    os <<              "ambientIntensity 0"<< endl;
    os <<              "diffuseColor 0.0352941 0.635294 0.666667"<< endl;
    os <<              "specularColor 0 0 0"<< endl;
    os <<              "shininess 0.78125"<< endl;
    os <<              "transparency 0.54"<< endl;
    os <<              "}"<< endl;
    os <<            "}"<< endl;
    os <<          "geometry IndexedFaceSet {"<< endl;
    os <<           "solid FALSE"<< endl;
    os <<           "coord DEF VTKcoordinates Coordinate {"<< endl;
    os <<           " point ["<< endl;
    os <<             points[0][0]<<" "<<points[0][1]<<" "<<points[0][2]<<","<< endl;
    os <<             points[1][0]<<" "<<points[1][1]<<" "<<points[1][2]<<","<< endl;
    os <<             points[2][0]<<" "<<points[2][1]<<" "<<points[2][2]<<","<< endl;
    os <<             points[3][0]<<" "<<points[3][1]<<" "<<points[3][2]<<","<< endl;
    os <<              "]"<< endl;
    os <<         " }"<< endl;
    os <<          " coordIndex  ["<< endl;
    os <<             "0, 1, 3, 2, -1,"<< endl;
    os <<           "]"<< endl;
    os <<         "}"<< endl;
    os <<         "}"<< endl;

}

void writeBbox(string name, vector<Vector3f> points, bool append=false)
{
    using namespace std;
    std::ofstream os;

    os.open(name.c_str(), append ? ios::app : ios::out);
    os.precision(15);

    if (!append) os <<"#VRML V2.0 utf8" << endl;
    os <<"Shape {"<<endl;
    os <<         " appearance Appearance {"<< endl;
    os <<            "material Material {"<< endl;
    os <<              "ambientIntensity 0"<< endl;
    os <<              "diffuseColor 0.0352941 0.635294 0.666667"<< endl;
    os <<              "specularColor 0 0 0"<< endl;
    os <<              "shininess 0.78125"<< endl;
    os <<              "transparency 0.4"<< endl;
    os <<              "}"<< endl;
    os <<            "}"<< endl;
    os <<          "geometry IndexedFaceSet {"<< endl;
    os <<           "solid FALSE"<< endl;
    os <<           "coord DEF VTKcoordinates Coordinate {"<< endl;
    os <<           " point ["<< endl;
    for(int i=0;i<points.size();++i)
        os <<             points[i][0]<<" "<<points[i][1]<<" "<<points[i][2]<<","<< endl;

    os <<              "]"<< endl;
    os <<         " }"<< endl;
    os <<          " coordIndex  ["<< endl;
    os <<             "0, 1, 3, 2, -1,"<< endl;
    os <<             "4, 6, 7, 5, -1,"<< endl;
    os <<             "8, 10, 11, 9, -1,"<< endl;
    os <<             "12, 13, 15, 14, -1,"<< endl;
    os <<             "16, 18, 19, 17, -1,"<< endl;
    os <<             "20, 21, 23, 22, -1,"<< endl;

    os <<           "]"<< endl;
    os <<         "}"<< endl;
    os <<         "}"<< endl;

}



template<typename VecA, typename VecB>
inline void
projectPoint(VecA const& P, VecB const& pt, VecB &projPt)
{
    VecB n;
    double d;
    d=P[P.size()-1];
    for(int i = 0; i < P.size()-1; ++i)
      n[i]=P[i];
    double dist=n.dot(pt)+d;
    n=dist*n;
    projPt=pt-n;
}


template<typename VecA, typename VecB>
inline void
reflectPoint(VecA const& P, VecB const& pt, VecB &projPt)
{
    VecB n;
    double d;
    d=P[P.size()-1];
    for(int i = 0; i < P.size()-1; ++i)
      n[i]=P[i];
    double dist=n.dot(pt)+d;
    n=2*dist*n;
    projPt=pt-n;
}

template<typename VecA, typename VecB>
inline void
reflectDirection(VecA const& P, VecB const& pt, VecB &projPt)
{
    VecB n;
    for(int i = 0; i < P.size()-1; ++i)
      n[i]=P[i];
    double dist=n.dot(pt);
    n=2*dist*n;
    projPt=pt-n;
}

template<typename Vec>
inline bool
vectorSimilarity(Vec const& A, Vec const& B, double threshold )
{
    double co = A.dot(B);
    double  na=A.norm();
    double nb=B.norm();
    co/=(na*nb);
    double ratio = (na>nb) ? nb/na : nb/na;
    return (ratio>threshold)&&(co>threshold);
}
Vector4f planeThroughPoints(Vector3f p1, Vector3f p2)
{
  //estimate plane from the 2 sampled points
  Vector4f P;
  Vector3f normal = p1-p2;
  Vector3f point = 0.5 * normal;
  point=p1+point;
  normal.normalize();
  double c = normal.dot(point);
  c=-c;
  P[0] = normal[0]; P[1] = normal[1]; P[2] = normal[2];
  P[3]=c;
  return P;
}


struct scoredPlane
{
  vector<Vector3f> p;
  double dim;
  double r;
};

bool better(const scoredPlane &a, const scoredPlane &b)
{
  return a.r>=b.r;
}


int scoreSymmetryPlane(vector<Vector3f> &points, Vector4f P, float radius,
                          std::vector<int> &inliers,
                          Vector3f boxPos, double maxDist, cv::flann::Index &kdtree)
{
  inliers.clear();
  cv::flann::SearchParams params;

    double distTotal=0;
    for(int i=0; i<points.size(); i++)
    {
      Vector3f q;
      reflectPoint(P,points[i],q);
      if(inRange(q,boxPos,maxDist))
      {

        vector<float> query;
        query.push_back(q[0]);
        query.push_back(q[1]);
        query.push_back(q[2]);

        vector<int> indices(1,0);
        vector<float> dists(1,0);
        int n=kdtree.radiusSearch(query, indices, dists, radius, 1,params);
        if(n>0)
        {
            distTotal+=dists[0];
            inliers.push_back(i);
        }

      }
    }

    return inliers.size();
}

inline int ransacRuns(int s, double ratio, double conf)
{
    double aux = pow(ratio,s);
    double t = log(1-conf)/log(1-aux);
    return t;
}

inline int ransacRunsSpecial(int N, int inlierSize, double conf)
{
    double aux = double(inlierSize)/(N*N);
    double t = log(1-conf)/log(1-aux);
    return t;
}

double
angleBetweenPlanes(Vector3f const& v, Vector3f const& w)
{
   double dot = v.dot(w) / v.norm() / w.norm();
   if(dot<0) dot=-dot;
   if (dot >= 1.0) return 0;
   return acos(dot);
}

double
planeDistance(Vector4f const& p1, Vector4f const& p2, bool &intersect)
{
    Vector3f norm1(p1[0],p1[1],p1[2]),norm2(p2[0],p2[1],p2[2]);

    double ang=angleBetweenPlanes(norm1,norm2);
    if(ang>0)
    {
        intersect=true;
        return ang;
    }
    Vector3f point(0,0,0);
    if(p1[0]!=0)
      point[0]=-p1[3]/p1[0];
    else
      point[1]=-p1[3]/p1[1];
    double dist = abs(point.dot(norm2)+p2[3]);
    intersect=false;
    return dist;
}

double computeSymmetry(vector<float> const &verts, vector<int> const &validIds,
                       Vector4f &P, std::vector<int> &inliers, double inlierThreshold,
                       vector<Vector4f> &planes, vector< vector<int> > &ins,
                       int Nplanes, Vector3f boxPos, double planeThresh, double maxDist,
                       cv::flann::Index &kdtree)
{
  int N = validIds.size();
  vector<Vector3f> pointsInBB;
  for(int i=0; i<N; ++i)
  {
    int ind = validIds[i];
    Vector3f pt(verts[ind], verts[ind+1], verts[ind+2]);
    pointsInBB.push_back(pt);
  }

  vector<int> indices(N);
  iota(indices.begin(), indices.end(),0);
  random_shuffle(indices.begin(), indices.end());
  int nRuns = 1;
  int sampleOffset = 0;
  bool badSample=false;
  int maxCount = 1000;
  Vector3f p1 = pointsInBB[indices[sampleOffset]] ;
  Vector3f p2 = pointsInBB[indices[sampleOffset+1]];
  int count = 0;
  do
  {
      p1 = pointsInBB[indices[sampleOffset]] ;
      p2 = pointsInBB[indices[sampleOffset+1]];
      Vector3f n = p1-p2;
      if(n[2]!=0 /*|| norm_L2(n)>maxDist*/)
      {
          badSample=true;
          sampleOffset+=2;
          ++count;
          if (sampleOffset + 1 >= N)
          {
              random_shuffle(indices.begin(), indices.end());
              sampleOffset = 0;
          }
      }
      else
          badSample=false;
  }while(badSample && count < maxCount);
  if(count == maxCount)
    return 0;
  count = 0;
  float inlierRatio;
  int expectedRuns, remainingRuns;
  P = planeThroughPoints(p1, p2);
  double const confidence = 0.95;

  int score = scoreSymmetryPlane(pointsInBB,P,inlierThreshold,
                                 inliers, boxPos,maxDist, kdtree);
//    planes.push(P);
//    ins.push(inliers);
  planes[0]=P;
  ins[0]=inliers;

  inlierRatio = float(inliers.size()) / N;
  expectedRuns = min(ransacRuns(2,0.05, confidence),1000);
  //expectedRuns= min(ransacRunsSpecial(N,N/3,confidence),1000);
  //expectedRuns = 10000;
 // cout<<"getinit: "<<expectedRuns<<endl;
//  cout<<"Inliers size: "<<inliers.size()<<endl;
  vector<int> curInliers;


  while (1)
  {
      remainingRuns = expectedRuns - nRuns;
      if (remainingRuns <= 0) break;
      Vector3f p1 = pointsInBB[sampleOffset] ;
      Vector3f p2 = pointsInBB[sampleOffset+1];
      count = 0;
      do
      {
        p1 = pointsInBB[sampleOffset] ;
        p2 = pointsInBB[sampleOffset+1];
          Vector3f n = p1-p2;
          if(n[2]!=0 /*|| norm_L2(n)>maxDist*/)
          {
            ++count;
              badSample=true;
              sampleOffset+=2;
              if (sampleOffset + 1 >= N)
              {
                  random_shuffle(indices.begin(), indices.end());
                  sampleOffset = 0;
              }
          }
          else
              badSample=false;
      }while(badSample && count < maxCount);
      if(count == maxCount)
        break;
      Vector4f P1 = planeThroughPoints(p1,p2);

      int score = scoreSymmetryPlane(pointsInBB,P1,inlierThreshold,
                                     curInliers, boxPos,maxDist, kdtree);
      ++nRuns;
    //  cout << "nRuns = " << nRuns << ", score = " << score << ", inliers.size() = " << inliers.size() << endl;
      sampleOffset += 2;
//        if(nRuns%1000==0){
//            cout << "nRuns = " << nRuns << ", expectedRuns = " << expectedRuns << ", inliers.size() = " << inliers.size() << " plane: " << endl;
//            displayVector(P);
//        }
      bool samePrev=false;
      for(int i=0;i<Nplanes;++i)
      {
          P=planes[i];
          bool intersect;
          double dist = planeDistance(P,P1,intersect);
          bool same;
          if(intersect)
              same=(dist<0.1);
          else
              same=(dist<planeThresh);
          if(same)
          {
              if(!samePrev)
              {
                  if(curInliers.size()>ins[i].size())//same and !
                  {
                      planes[i]=P1;
                      ins[i]=curInliers;
                      expectedRuns= min(ransacRuns(2, (double)inliers.size()/N,confidence),1000);

                  }
                  samePrev=true;
              }
              else
              {
                  planes.erase(planes.begin()+i);
                  ins.erase(ins.begin()+i);
                  planes.push_back(Vector4f(0,0,0,0));
                  vector<int> aux;
                  ins.push_back(aux);
              }
          }
          else
          {
              if(curInliers.size()>ins[i].size())//different and better! we insert it and say we already saw it
              {
                  planes.insert(planes.begin()+i,P1);
                  ins.insert(ins.begin()+i,curInliers);
                  planes.pop_back();
                  ins.pop_back();
                  samePrev=true;
                  expectedRuns= min(ransacRuns(2, (double)inliers.size()/N,confidence),1000);

              }

          }

      }
      inliers=ins[0];
      P=planes[0];

  } // end while
  return double(inliers.size()) / N;

}

void write_ply_file(const std::string & filename, vector<float> &verts,  vector<int32_t> &vertexIndices)
{

    std::filebuf fb;
    fb.open(filename, std::ios::out | std::ios::binary);
    std::ostream outputStream(&fb);

    PlyFile myFile;

    myFile.add_properties_to_element("vertex", { "x", "y", "z" }, verts);
    //  myFile.add_properties_to_element("vertex", { "nx", "ny", "nz" }, norms);

    // List property types must also be created with a count and type of the list (data property type
    // is automatically inferred from the type of the vector argument).
    myFile.add_properties_to_element("face", { "vertex_indices" }, vertexIndices, 3, PlyProperty::Type::UINT8);

    myFile.comments.push_back("generated by tinyply");
    myFile.write(outputStream, true);

    fb.close();
}

void read_ply_file(const std::string & filename,vector<float> &verts,  vector<int32_t> &faces)
{
    // Tinyply can and will throw exceptions at you!
    try
    {
        // Read the file and create a std::istringstream suitable
        // for the lib -- tinyply does not perform any file i/o.
        std::ifstream ss(filename, std::ios::binary);

        // Parse the ASCII header fields
        PlyFile file(ss);

        for (auto e : file.get_elements())
        {
            std::cout << "element - " << e.name << " (" << e.size << ")" << std::endl;
            for (auto p : e.properties)
            {
                std::cout << "\tproperty - " << p.name << " (" << PropertyTable[p.propertyType].str << ")" << std::endl;
            }
        }
        std::cout << std::endl;

        for (auto c : file.comments)
        {
            std::cout << "Comment: " << c << std::endl;
        }

        // Define containers to hold the extracted data. The type must match
        // the property type given in the header. Tinyply will interally allocate the
        // the appropriate amount of memory.


        uint32_t vertexCount, normalCount, colorCount, faceCount, faceTexcoordCount, faceColorCount;
        vertexCount = normalCount = colorCount = faceCount = faceTexcoordCount = faceColorCount = 0;

        // The count returns the number of instances of the property group. The vectors
        // above will be resized into a multiple of the property group size as
        // they are "flattened"... i.e. verts = {x, y, z, x, y, z, ...}
        vertexCount = file.request_properties_from_element("vertex", { "x", "y", "z" }, verts);
        //  normalCount = file.request_properties_from_element("vertex", { "nx", "ny", "nz" }, norms);

        // For properties that are list types, it is possibly to specify the expected count (ideal if a
        // consumer of this library knows the layout of their format a-priori). Otherwise, tinyply
        // defers allocation of memory until the first instance of the property has been found
        // as implemented in file.read(ss)
        faceCount = file.request_properties_from_element("face", { "vertex_indices" }, faces, 3);

        // Now populate the vectors...
        file.read(ss);

        // Good place to put a breakpoint!
        std::cout << "\tRead " << verts.size() << " total vertices (" << vertexCount << " properties)." << std::endl;
        // std::cout << "\tRead " << norms.size() << " total normals (" << normalCount << " properties)." << std::endl;
        std::cout << "\tRead " << faces.size() << " total faces (triangles) (" << faceCount << " properties)." << std::endl;


    }

    catch (const std::exception & e)
    {
        std::cerr << "Caught exception: " << e.what() << std::endl;
    }
}
struct aabb
{
    float minX,maxX,minY,maxY,minZ,maxZ;
};


bool intersect(aabb a, aabb b) {
    return (a.minX <= b.maxX && a.maxX >= b.minX) &&
            (a.minY <= b.maxY && a.maxY >= b.minY) &&
            (a.minZ <= b.maxZ && a.maxZ >= b.minZ);
}

aabb computeBbox(const vector<float>  &verts)
{
    vector<float> x,y,z;
    aabb bb;
    for(int i=0; i<verts.size(); i+=3)
    {
        Vector3f v(verts[i],verts[i+1],verts[i+2]);
        x.push_back(v[0]);
        y.push_back(v[1]);
        z.push_back(v[2]);
    }
    bb.maxX=*max_element(x.begin(),x.end());
    bb.maxY=*max_element(y.begin(),y.end());
    bb.maxZ=*max_element(z.begin(),z.end());

    bb.minX=*min_element(x.begin(),x.end());
    bb.minY=*min_element(y.begin(),y.end());
    bb.minZ=*min_element(z.begin(),z.end());
    return bb;
}
aabb computeBbox(const vector<Vector3f>  &verts)
{
    vector<float> x,y,z;
    aabb bb;
    for(int i=0; i<verts.size(); i++)
    {
        Vector3f v = verts[i];
        x.push_back(v[0]);
        y.push_back(v[1]);
        z.push_back(v[2]);
    }
    bb.maxX=*max_element(x.begin(),x.end());
    bb.maxY=*max_element(y.begin(),y.end());
    bb.maxZ=*max_element(z.begin(),z.end());

    bb.minX=*min_element(x.begin(),x.end());
    bb.minY=*min_element(y.begin(),y.end());
    bb.minZ=*min_element(z.begin(),z.end());
    return bb;
}

std::string extractBaseFileName(const std::string &fullFileName)
{
    size_t pos = fullFileName.find_last_of('/');
    std::string baseName;
    if (pos != std::string::npos)
    {
        baseName = fullFileName.substr(pos+1, fullFileName.size()-pos);
    }
    else
    {
        baseName = fullFileName;
    }
    // remove the ending
    pos = baseName.find_last_of('.');
    if (pos != std::string::npos)
    {
        return baseName.substr(0, pos);
    }
    else
    {
        return baseName;
    }
}


void saveTxtPlane(const string &name, const vector<vector<Vector3f> > &planes, const vector<float> &dims)
{
    ofstream os(name.c_str());
    os<<planes.size()<<endl;
    for(int i=0;i<planes.size();++i)
    {
        for(int j=0;j<3;++j)
            os<<planes[i][j][0]<<" "<<planes[i][j][1]<<" "<<planes[i][j][2]<<endl;
        os<<dims[i]<<endl;
    }
}

float getPlane(Vector4f P, const vector<float> &verts, const vector<int> &boxIds,
               const vector<int> &inliers, vector<Vector3f> &plane, double distMax)
{
  Vector3f n(P[0],P[1],P[2]);
  n.normalize();
  int N = boxIds.size();
  vector<Vector3f> pointsInBB;
  for(int i=0; i<N; ++i)
  {
    int ind = boxIds[i];
    Vector3f pt(verts[ind], verts[ind+1], verts[ind+2]);
    pointsInBB.push_back(pt);
  }
  vector<Vector3f> inPts;
  for(int i=0; i<inliers.size(); ++i)
    inPts.push_back(pointsInBB[inliers[i]]);
  Vector3f z(0,0,1);
  Vector3f x = n.cross(z);
  Matrix3f R;
  R << x[0], x[1], x[2],
       n[0], n[1], n[2],
       z[0], z[1], z[2];
  Vector3f p0(0,0,0);
  if(n[0]!=0)
    p0[0] = - P[3]/n[0];
  else
    p0[1] = - P[3]/n[1];
  Vector3f T = -R*p0;
  aabb bbox1 = computeBbox(inPts);
  double dx=bbox1.maxX-bbox1.minX,dy=bbox1.maxY-bbox1.minY,dz=bbox1.maxZ-bbox1.minZ;
  if(dx<distMax/2 || dy <distMax/2 || dz <distMax/2)
    return 0;
  for(int i=0; i<inPts.size(); ++i)
    inPts[i] = R*inPts[i]+T;

  aabb bbox = computeBbox(inPts);
  float dim = bbox.maxY - bbox.minY;
  Vector3f v(bbox.minX,0,bbox.minZ);
  plane.resize(4);
  plane[0]=v;
  v[2]=bbox.maxZ;
  plane[1]=v;
  v[0]=bbox.maxX;
  plane[3]=v;
  v[2]=bbox.minZ;
  plane[2]=v;
  Matrix3f Rt = R.transpose();
  for(int i=0; i<plane.size(); ++i)
    plane[i] = Rt*plane[i]+p0;
  return dim;
}

void cleanVerts(vector<float> &verts, const vector<int32_t> &faces)
{
  vector<float> newVerts;
  vector<bool> present(verts.size()/3,false);
  for(int i=0; i<faces.size(); ++i)
    present[faces[i]]=true;
  for(int i=0; i<verts.size(); i+=3)
  {
    if(present[i/3])
    {
      newVerts.push_back(verts[i]);
      newVerts.push_back(verts[i+1]);
      newVerts.push_back(verts[i+2]);
    }
  }
  verts = newVerts;
}
} // end namespace <>


int
main(int argc, char * argv[])
{
  std::string modelList;
  std::string outputFolder;
  double maxDist, thresh;
  int nPlanes;
  boost::program_options::options_description desc("Allowed options");
  desc.add_options()
          ("help", "Produce help message")
          ("modelList", boost::program_options::value<std::string>(&modelList)->default_value("test.txt"), "Model file list")
          ("outputFolder", boost::program_options::value<std::string>(&outputFolder)->default_value("detectedSym"), "Output folder")
          ("threshold", boost::program_options::value<double>(&thresh)->default_value(0.1),"Threshold for similar planes")
          ("bboxSize", boost::program_options::value<double>(&maxDist)->default_value(6),"bbox size")
          ("nPlanes", boost::program_options::value<int>(&nPlanes)->default_value(2),"number of planes per box")

          ;

  boost::program_options::variables_map vm;
  boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).run(), vm);
  boost::program_options::notify(vm);

  if (vm.count("help"))
  {
      std::cout << desc << std::endl;
      return 1;
  }

  if (!boost::filesystem::exists(outputFolder))
  {
      if (!boost::filesystem::create_directory(outputFolder))
      {
          cout<<"Error creating output directory."<<endl;
          return 0;
      }
  }

  std::ifstream modelStream;
  modelStream.open(modelList.c_str());

  if (!modelStream.is_open())
  {
      cout<<"Could not open images file list"<<endl;
      return 0;
  }

  std::vector<std::string> modelFileNames;
  std::string fileName;
  while (modelStream >> fileName)
        modelFileNames.push_back(fileName);


  cout<<modelFileNames.size()<<endl;

    srand(unsigned(time(0)));

    for (unsigned int i = 0; i < modelFileNames.size(); i++)
    {
        string fileName=modelFileNames[i];

        vector<float> verts;
        vector<int32_t> faces;

        string baseName = extractBaseFileName(fileName);
        string outName=outputFolder+"/"+baseName+".txt";
        string outName2=outputFolder+"/"+baseName+".wrl";


        cout<<"Processing: "<<baseName<<endl;
        read_ply_file(fileName,verts,faces);

        cleanVerts(verts,faces);

        aabb bbox = computeBbox(verts);

    int N = verts.size()/3;
    double proportion = maxDist/(bbox.maxX-bbox.minX);
    int minInliers = 10;
    cout<<"min numb of box points "<<10<<endl;

    vector<vector<Vector3f> > totalPlanes;
    vector<float> totalDims;
    vector<scoredPlane> allPlanes;
    int count = 0;
    vector<cv::Point3f> pointsOr;

    for(int i=0; i<verts.size(); i+=3)
    {
        cv::Point3f p(verts[i],verts[i+1],verts[i+2]);
        pointsOr.push_back(p);

    }

    cv::flann::KDTreeIndexParams indexParams;
    cv::flann::Index kdtree(cv::Mat(pointsOr).reshape(1), indexParams);
//#pragma omp parallel for
    for(double maxDist = 4; maxDist<20; ++maxDist)
    {
      cout<<"============================= BOX SIZE: "<<maxDist<<"================================="<<endl;
    for(int z=bbox.minZ;z<bbox.maxZ-maxDist;z+=maxDist/2)
    {
        for(int y=bbox.minY;y<bbox.maxY-maxDist;y+=maxDist/2)
            for(int x=bbox.minX;x<bbox.maxX-maxDist;x+=maxDist/2)
            {
              //cout<<"box "<<++count<<endl;
                vector<int> boxIds;
                for(int i=0;i<verts.size();i+=3)
                {
                  if(verts[i] >= x && verts[i] < x+maxDist &&
                     verts[i+1] >= y && verts[i+1] < y+maxDist &&
                     verts[i+2] >= z && verts[i+2] < z+maxDist)
                    boxIds.push_back(i);
                }
                if(boxIds.size()<minInliers)
                    continue;

                Vector3f pos(x,y,z);
                Vector4f P;
                vector<int> inliers;
                vector<Vector4f> planes;
                vector< vector<int> > ins;

                for(int i=0;i<nPlanes;++i)
                {
                    vector<int> crap;
                    ins.push_back(crap);

                    Vector4f aux(0,0,0,0);
                    planes.push_back(aux);
                }
                double radius = maxDist/20;
                computeSymmetry(verts,boxIds,P,inliers,radius,planes,ins,
                                nPlanes,pos, thresh, maxDist,kdtree);

                int n=planes.size();

                for(int j=0;j<n;++j)
                {
                    vector<int> inliers=ins[j];
                    double ratio = double(inliers.size())/boxIds.size();
                    if(ratio<0.3)
                        continue;

                    vector<Vector3f> P;
                    float dim = getPlane(planes[j],verts,boxIds,
                                                     inliers, P,maxDist);
                    if(dim == 0)
                      continue;
                    cout<<"Symmetry plane found with "<<inliers.size()<<" inliers and "<<ratio<<" inlier ratio at: "<<endl;

                    Vector4f pp=planes[j];
                    Vector3f n(pp[0],pp[1],pp[2]);
//                    cout<<"plane found "<<planes[j]<<endl;
//                    for(int k=0;k<4;++k)
//                    {
//                      double dist = n.dot(P[k])+pp[3];
//                      cout<<"dist "<<dist<<endl;
//                    }

                    //totalDims.push_back(dim);
                    //totalPlanes.push_back(P);
                    scoredPlane sp;
                    sp.p = P;
                    sp.dim = dim;
                    sp.r = ratio;
                    allPlanes.push_back(sp);

                }
            }
    }
    }
    std::sort(allPlanes.begin(),allPlanes.end(),better);
    for(int i=0; i<min((int)allPlanes.size(),50); ++i)
    {
      totalDims.push_back(allPlanes[i].dim);
      totalPlanes.push_back(allPlanes[i].p);
    }
    for(int i=0;i<totalPlanes.size();++i)
        writePlane(outName2,totalPlanes[i], i!=0);
    saveTxtPlane(outName,totalPlanes,totalDims);
  }

    return 0;
} // end main()




