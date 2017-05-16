#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <random>
#include <algorithm>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include  <eigen3/Eigen/Dense>
#include  <eigen3/Eigen/Geometry>
#include "tinyply.h"

using namespace tinyply;
using namespace std;
using namespace Eigen;

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

void write_ply_file(const std::string & filename, vector<float> &verts,  vector<int32_t> &vertexIndices)
{

    std::filebuf fb;
    fb.open(filename, std::ios::out | std::ios::binary);
    std::ostream outputStream(&fb);

    PlyFile myFile;

    myFile.add_properties_to_element("vertex", { "x", "y", "z" }, verts);
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


        uint32_t vertexCount, faceCount;
        vertexCount = faceCount = 0;

        // The count returns the number of instances of the property group. The vectors
        // above will be resized into a multiple of the property group size as
        // they are "flattened"... i.e. verts = {x, y, z, x, y, z, ...}
        vertexCount = file.request_properties_from_element("vertex", { "x", "y", "z" }, verts);

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
float randomAngle()
{
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(0,2*M_PI);
    return dis(gen);
}

float randomFloat(float a, float b)
{
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(a,b);
    return dis(gen);
}

Vector3f randomAxis()
{
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(-1, nextafter(1,numeric_limits<float>::max()));
    float u=dis(gen);
    float theta=randomAngle();
    float x=sqrt(1-u*u)*cos(theta);
    float y=sqrt(1-u*u)*sin(theta);
    float z=u;
    Vector3f axis(x,y,z);
    return axis;
}


void rotate(vector<float> &verts, vector<vector<Vector3f> > &planes)
{
    Matrix3f m;
    Vector3f axis=randomAxis();
    float alpha=randomAngle();
    m=AngleAxisf(alpha, axis);
    for(int i=0; i<verts.size(); i+=3)
    {
        Vector3f v(verts[i],verts[i+1],verts[i+2]);
        v=m*v;
        verts[i]=v[0];
        verts[i+1]=v[1];
        verts[i+2]=v[2];

    }
    for(int i=0;i<planes.size();++i)\
    {
        for(int j=0;j<4;++j)
            planes[i][j]=m*planes[i][j];
    }


}

void rotateZ(vector<float> &verts, vector<vector<Vector3f> > &planes)
{
    Matrix3f m;
    float alpha=randomAngle();
    m=AngleAxisf(alpha, Vector3f::UnitZ());
    for(int i=0; i<verts.size(); i+=3)
    {
        Vector3f v(verts[i],verts[i+1],verts[i+2]);
        v=m*v;
        verts[i]=v[0];
        verts[i+1]=v[1];
        verts[i+2]=v[2];

    }
    for(int i=0;i<planes.size();++i)\
    {
        for(int j=0;j<4;++j)
            planes[i][j]=m*planes[i][j];
    }


}

void scale(vector<float> &verts, vector<vector<Vector3f> > &planes, vector<float> dims)
{
    float size=randomFloat(3,10);
    float maxDim=0;
    for(int i=0;i<planes[0].size()-1;++i)
    {
        float d;
        for(int j=0;j<3;++j)
        {
            d=abs(planes[0][i+1][j]-planes[0][i][j]);
            maxDim=max(maxDim,d);
        }
    }
    maxDim=max(maxDim,dims[0]);
    if(maxDim==0)
        return;
    float m=size/maxDim;
    for(int i=0; i<verts.size(); i+=3)
    {
        Vector3f v(verts[i],verts[i+1],verts[i+2]);
        v=m*v;
        verts[i]=v[0];
        verts[i+1]=v[1];
        verts[i+2]=v[2];

    }
    for(int i=0;i<planes.size();++i)\
    {
        for(int j=0;j<4;++j)
            planes[i][j]=m*planes[i][j];
    }
    for(int i=0;i<dims.size();++i)
        dims[i]=m*dims[i];

}

void scale(vector<float> &verts, vector<vector<Vector3f> > &planes, vector<float> dims, float m)
{

    for(int i=0; i<verts.size(); i+=3)
    {
        Vector3f v(verts[i],verts[i+1],verts[i+2]);
        v=m*v;
        verts[i]=v[0];
        verts[i+1]=v[1];
        verts[i+2]=v[2];

    }
    for(int i=0;i<planes.size();++i)\
    {
        for(int j=0;j<4;++j)
            planes[i][j]=m*planes[i][j];
    }
    for(int i=0;i<dims.size();++i)
        dims[i]=m*dims[i];

}

void writePlane(const string &name, const vector<Vector3f> &points, bool append=false)
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

void writeBbox(const string &name, const vector<Vector3f> &points, bool append=false)
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

void readTxtPlane(const string &name, vector<vector<Vector3f> > &planes, vector<float> &dims)
{
    ifstream is(name.c_str());
    int n;
    is >> n;
    for(int i=0;i<n;++i)
    {
        vector<Vector3f> p(4);
        float d;
        for(int j=0;j<3;++j)
            is >> p[j][0] >> p[j][1] >> p[j][2];
        is >> d;
        planes.push_back(p);
        dims.push_back(d);
    }
    for(int l=0; l<planes.size(); ++l)
    {
        vector<Vector3f> p=planes[l];
        int i;
        for(int j=0;j<3;++j)
            if(p[0][j]==0 && p[1][j]==0 && p[2][j]==0)
                i=j;

        int j=(i+1)%3;
        int k=(i+2)%3;
        p[3][i]=0;        p[3][j]=p[2][j];        p[3][k]=-p[2][k];
        planes[l]=p;
    }
}

void translate(vector<Vector3f>& points, Vector3f T)
{
    for(int i=0;i<points.size();++i)
        points[i]=points[i]+T;
}

void translateXY(vector<float> &verts, vector<vector<Vector3f> > &planes, Vector3f T)
{

    for(int i=0; i<verts.size(); i+=3)//center
    {
        Vector3f v(verts[i],verts[i+1],verts[i+2]);
        v=v+T;

        verts[i]=v[0];
        verts[i+1]=v[1];
        verts[i+2]=v[2];
    }
    for(int i=0;i<planes.size();++i)
        translate(planes[i],T);
}


void translateXY(vector<float> &verts, vector<vector<Vector3f> > &planes)
{
    float x = randomFloat(-4,4);
    float y = randomFloat(-4,4);
    Vector3f T(x,y,0);
    for(int i=0; i<verts.size(); i+=3)//center
    {
        Vector3f v(verts[i],verts[i+1],verts[i+2]);
        v=v+T;

        verts[i]=v[0];
        verts[i+1]=v[1];
        verts[i+2]=v[2];
    }
    for(int i=0;i<planes.size();++i)
        translate(planes[i],T);
}

void translateZ(vector<float> &verts, vector<vector<Vector3f> > &planes)
{
    float z=1000;

    for(int i=0; i<verts.size(); i+=3)
        z=min(z,verts[i+2]);
    for(int i=0; i<verts.size(); i+=3)
        verts[i+2]-=z;
    Vector3f T(0,0,-z);
    for(int i=0;i<planes.size();++i)
        translate(planes[i],T);
}
void translateZ2(vector<float> &verts, vector<vector<Vector3f> > &planes)
{
    float z=-1000;

    for(int i=0; i<verts.size(); i+=3)
        z=max(z,verts[i+2]);
    for(int i=0; i<verts.size(); i+=3)
        verts[i+2]-=z;
    Vector3f T(0,0,-z);
    for(int i=0;i<planes.size();++i)
        translate(planes[i],T);
}
void randomModify(vector<float> &verts, vector<vector<Vector3f> > &planes, vector<float> &dims)
{



    scale(verts,planes,dims);
    rotateZ(verts,planes);
    translateZ(verts,planes);

}

void createScene(vector<string> const& modelFileNames, vector<float> &totalVerts, vector<int32_t> &totalFaces,
                 vector<vector<Vector3f> > &totalPlanes, vector<float> &totalDims,
                  int nObjects, string inputFolder, string outputFolder, int j)
{
    vector<aabb> bboxes;
    for (unsigned int i = 0; i < nObjects; i++)
    {
        string baseName=modelFileNames[i];
        vector<float> verts;
        vector<int32_t> faces;
        vector<vector<Vector3f> > planes;
        vector<float> dims;

        cout<<"Processing: "<<baseName<<endl;
        read_ply_file(inputFolder+"/"+baseName+".ply",verts,faces);

        readTxtPlane(inputFolder+"/"+baseName+"-plane.txt",planes,dims);

        cout<<"Read "<<planes.size()<<" planes"<<endl;

        randomModify(verts,planes,dims);\

        aabb bb=computeBbox(verts);
        bool collision=false;
        do{
            collision=false;
            for(int k=0;k<bboxes.size();++k)
            {
                if(intersect(bboxes[k],bb))
                {
                    collision=true;
                    break;
                }
            }
            if(collision)
            {
                translateXY(verts,planes);
                bb=computeBbox(verts);
            }
        }while(collision);
        bboxes.push_back(bb);

        for(int k=0;k<faces.size();++k)
            faces[k]=faces[k]+totalVerts.size()/3;

        totalVerts.insert(totalVerts.end(),verts.begin(),verts.end());
        totalFaces.insert(totalFaces.end(),faces.begin(),faces.end());

        totalPlanes.insert(totalPlanes.end(),planes.begin(),planes.end());
        totalDims.insert(totalDims.end(),dims.begin(),dims.end());
        stringstream planeName;
        planeName<<outputFolder<<"/"<<j<<"-scene-plane.wrl";
        for(int k=0;k<planes.size();++k)
            writePlane(planeName.str(),planes[k],i!=0||k!=0);

    }
    //add table

    vector<float> dimsScene(2);
    float minx=0,miny=0,maxx=0,maxy=0;
    for(int i=0;i<bboxes.size();++i)
    {
        minx=min(bboxes[i].minX,minx);
        miny=min(bboxes[i].minY,miny);
        maxx=max(bboxes[i].maxX,maxx);
        maxy=max(bboxes[i].maxY,maxy);
    }
    dimsScene[0]=maxx-minx;
    dimsScene[1]=maxy-miny;
    Vector3f center((minx+maxx)/2.0,(miny+maxy)/2.0,0);

    vector<float> verts;
    vector<int32_t> faces;
    vector<vector<Vector3f> > planes;
    vector<float> dims;
    read_ply_file("table.ply",verts,faces);
    readTxtPlane("table-plane.txt",planes,dims);

    float scalex,scaley,m;
    scalex=dimsScene[0]/dims[0];
    scaley=dimsScene[1]/dims[1];
    m=max(scalex,scaley);

    scale(verts,planes,dims,m);
    translateZ2(verts,planes);
    translateXY(verts,planes,center);

    stringstream planeName;
    planeName<<outputFolder<<"/"<<j<<"-scene-plane.wrl";
    for(int k=0;k<planes.size();++k)
        writePlane(planeName.str(),planes[k],true);

    for(int k=0;k<faces.size();++k)
        faces[k]=faces[k]+totalVerts.size()/3;

    totalVerts.insert(totalVerts.end(),verts.begin(),verts.end());
    totalFaces.insert(totalFaces.end(),faces.begin(),faces.end());

    totalPlanes.insert(totalPlanes.end(),planes.begin(),planes.end());
    totalDims.insert(totalDims.end(),dims.begin(),dims.end());
}


int main(int argc, char *argv[])
{
    std::string modelList;
    std::string outputFolder;
    std::string inputFolder;
    int n;
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
            ("help", "Produce help message")
            ("modelList", boost::program_options::value<std::string>(&modelList)->default_value("train.txt"), "Axis aligned model file list")
            ("numberScenes", boost::program_options::value<int>(&n)->default_value(200), "Number of scenes to generate")
            ("inputFolder", boost::program_options::value<std::string>(&inputFolder)->default_value("aligned"), "Input folder where the axis aligned models are located")
            ("outputFolder", boost::program_options::value<std::string>(&outputFolder)->default_value("localScene"), "Output folder")
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
        cout<<"Could not open models file list"<<endl;
        return 0;
    }

    std::vector<std::string> modelFileNames;
    std::string fileName;
    while (modelStream >> fileName)
    {
        modelFileNames.push_back(fileName);
    }

    //initialize uniform int distributed generator
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(3,10);

    for(int j=0; j<n;++j)
    {

        int nObjects=dis(gen);
        //int nObjects=2;
        shuffle(modelFileNames.begin(),modelFileNames.end(),gen);
        vector<float> totalVerts;
        vector<int32_t> totalFaces;
        vector<vector<Vector3f> > totalPlanes;
        vector<float> totalDims;

        createScene(modelFileNames,totalVerts,totalFaces,totalPlanes,totalDims, nObjects, inputFolder, outputFolder, j);

        stringstream outName;
        outName<<outputFolder<<"/"<<j<<"-scene.ply";
        write_ply_file(outName.str(),totalVerts,totalFaces);
        stringstream outName2;
        outName2<<outputFolder<<"/"<<j<<"-scene-plane.txt";
        saveTxtPlane(outName2.str(), totalPlanes,totalDims);
    }
    return 0;
}