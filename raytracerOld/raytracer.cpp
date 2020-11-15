#include <iostream>
#include "parser.h"
#include "ppm.h"


#include <cmath> // MAKE SURE THAT IT IS ALLOWED



typedef unsigned char RGB[3];

//using namespace parser;

////////////////we may want to use struct instead of class

struct Vec3f // Is ": parser::Vec3f" necesssary?
{

    float x, y, z;

    Vec3f(){
        //printf("\n empty constructor \n");
    }

    Vec3f(parser::Vec3f vector) : x(vector.x), y(vector.y), z(vector.z) {

    }
    Vec3f(float x, float y, float z) : x(x), y(y), z(z) {}
    
    Vec3f operator * (float d) const { 
        //printf("Distance: %lf\n", d );
        //printf("MULTIPLICATION\n");
        return Vec3f(x*d, y*d, z*d); 
    }

    Vec3f operator + (Vec3f v) const { 
        return Vec3f(x+v.x, y+v.y, z+v.z); 
    }

    Vec3f operator - (Vec3f v) const { 
        return Vec3f(x-v.x, y-v.y, z-v.z); 
    }

    Vec3f operator = (parser::Vec3f vector) const { 
        printf("Assignment! \n" );

        return Vec3f(vector); 
    }

    Vec3f operator-() const {
        Vec3f v;
        v.x = -x;
        v.y = -y;
        v.z = -z;
        return v;
   }

    Vec3f normalize() const {
        float norm = sqrt(x*x + y*y + z*z);
        return Vec3f(x/norm,y/norm,z/norm);
    }
    
};

class Ray{


    private:
        

    public:
        
        Vec3f e;
        Vec3f d;

        Ray(){
            //printf("empty ray constructor\n");
        }
        Ray(Vec3f origin, Vec3f direction ){

            //printf("ray constructor\n");


            e = origin;
            d = direction;
        }
        Vec3f RayVect(float t){

            Vec3f v2 = Vec3f(d.x*t, d.y*t, d.z*t);

            Vec3f result = Vec3f(e.x + v2.x, e.y + v2.y, e.z + v2.z );

            return result;
        }
       
        //~Ray();
    
};

Vec3f crossProduct(Vec3f u , Vec3f v ){

    Vec3f result = Vec3f( (u.y*v.z - u.z*v.y) , (u.z*v.x - u.x*v.z) , (u.x*v.y - u.y*v.x) );

    return result;

}

float dotProduct(Vec3f u , Vec3f v ){

    return (u.x*v.x + u.y*v.y + u.z*v.z);

}

Vec3f clamp(Vec3f vector) {
  Vec3f v ;
  v.x = (vector.x > 255) ? 255 : (vector.x < 0) ? 0 : vector.x;
  v.y = (vector.y > 255) ? 255 : (vector.y < 0) ? 0 : vector.y;
  v.z = (vector.z > 255) ? 255 : (vector.z < 0) ? 0 : vector.z;
  return v;
}

bool intersection(Ray ray, parser::Sphere sphere, Vec3f center , float& t, Vec3f& surfaceNormal){

    Vec3f e = ray.e;
    Vec3f d = ray.d;

    float r = sphere.radius; // radius of the sphere

    float a = dotProduct(d,d);           // a is A in the equation -> At^2 + Bt + C = 0 // 
    float b = 2*dotProduct(d,e-center);       // b is B in the equation -> At^2 + Bt + C = 0 // 
    float c = dotProduct(e-center,e-center) - r*r; // c is C in the equation -> At^2 + Bt + C = 0 // 

    float discriminant = b*b - 4*a*c;

    if (discriminant < 0.005) // 
    {
        return false;
    }
    else{
        float x0 = (-b - sqrt(discriminant))/(2*a); // one of the real roots of the equation
        float x1 = (-b + sqrt(discriminant))/(2*a); // one of the real roots of the equation
        t = (x0 < x1) ? x0 : x1;
        //printf("t1 %lf \n", x0 );
        //printf("t2 %lf \n", x1 );
        
        Vec3f pointOnTheSphere  = ray.e + ray.d*t; 

        surfaceNormal = (pointOnTheSphere - center) * (1.0 / sphere.radius);
        return true;        
    }

    //Vec3f c = sphere.vertex_data[scene.center_vertex_id]; // center of the sphere
}



bool intersection(Ray ray, parser::Face face, parser::Scene scene,  float& t, Vec3f& surfaceNormal){

    Vec3f e = ray.e; // origin 
    Vec3f d = ray.d; // direction

    Vec3f p ; // the ray-plane intersection point (may or may not be inside the triangle) 

    float gama, beta; // variables for barycentric coordinates


    Vec3f v1 = scene.vertex_data[face.v0_id - 1];
    Vec3f v2 = scene.vertex_data[face.v1_id - 1];
    Vec3f v3 = scene.vertex_data[face.v2_id - 1];

    // calculating plane normal


    Vec3f normalVector = crossProduct( v3-v2 , v2-v1);  // BE CAREFULL ABOUT THE ORDER OF THE VERTICES
    surfaceNormal = -normalVector; // TO BE USED BY SHADING PART OF THE CODE

    if (dotProduct(normalVector , d)  < 0.000001) // if plane and ray are parallel 
    {
        return false;
    }

    t = (dotProduct((v1 - e),normalVector))/(dotProduct(d,normalVector)); // calculating t to find the ray-plane intersection point "p"


    //printf("t : %lf \n" , t);

    p = e + d * t;


    //printf("TEST1\n");

    /*
    if (t <= 0.000001) // t_min
    {
        return false;
    }
    */

    //printf("TEST2\n");

    /////////////////////////////////////////////

    //calculating the barycentric coordanates
    

    /*

    https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates


    // Compute barycentric coordinates (u, v, w) for
    // point p with respect to triangle (a, b, c)
    void Barycentric(Point p, Point a, Point b, Point c, float &u, float &v, float &w)
    {
        Vector v0 = b - a, v1 = c - a, v2 = p - a;
        float d00 = Dot(v0, v0);
        float d01 = Dot(v0, v1);
        float d11 = Dot(v1, v1);
        float d20 = Dot(v2, v0);
        float d21 = Dot(v2, v1);
        float denom = d00 * d11 - d01 * d01;
        v = (d11 * d20 - d01 * d21) / denom;
        w = (d00 * d21 - d01 * d20) / denom;
        u = 1.0f - v - w;
    }

    */


    //a = v1 
    //b = v2 
    //c = v3 
    //v0 = v_21 
    //v1 = v_31 
    //v2 = v_p1 

    Vec3f v_21 = v2-v1;
    Vec3f v_31 = v3-v1;
    Vec3f v_p1 = p-v1;

    float p1 = dotProduct(v_21, v_21);
    float p2 = dotProduct(v_21, v_31);
    float p3 = dotProduct(v_31, v_31);
    float p4 = dotProduct(v_p1, v_21);
    float p5 = dotProduct(v_p1, v_31);


    float den = p1*p3 - p2*p2; // denominator

    gama = (p3*p4 - p2*p5) / den; // GAMA OR BETA ???

    //printf("GAMA : %lf \n", gama);

    if (gama < 0 || gama > 1 )
    {
        return false;
    }

    //printf("TEST3\n");


    beta = (p1*p5 - p2*p4) / den; // BETA OR GAMA ???

    if (beta < 0 || beta > 1-gama)
    {
        return false;
    }

    //printf("TEST4\n");



    return true;
}













bool isUnderShadow(Vec3f& pointOnTheMesh, Vec3f& vectorToLight, parser::Scene& scene, float& t, float& lightDistance, Vec3f& surfaceNormal){


    for (int i = 0; i < scene.spheres.size(); ++i)
    {
        Vec3f center = scene.vertex_data[scene.spheres[i].center_vertex_id-1]; // center of the sphere 
        
        Ray shadowRay = Ray(pointOnTheMesh,vectorToLight); 

        float t_to_sphere;

        Vec3f sphereSurfaceNormal; // = (pointOnTheSphere - center) * (1.0 / sphere.radius);

        if (intersection(shadowRay, scene.spheres[i], center ,t_to_sphere , sphereSurfaceNormal)){

            Vec3f point_of_Obstacle = pointOnTheMesh + shadowRay.d*t_to_sphere;
            Vec3f pointToObstacle = Vec3f(scene.point_lights[0].position) - point_of_Obstacle;


            if (sqrt(dotProduct(pointToObstacle,pointToObstacle)) < lightDistance)
            {
                return true;

            }
        }    
    }





}

Vec3f mirrorShader(){
        /*
    if (mirrorShadingParams.x != 0 || mirrorShadingParams.y != 0 || mirrorShadingParams.z != 0 )
    {
        printf("MIRROR\n");

        float t1,t2,t3;

        sphereSurfaceNormal = sphereSurfaceNormal.normalize();

        eyeRay.d = eyeRay.d.normalize();

        Vec3f mirrorReflactanceRayDirection  = eyeRay.d + (sphereSurfaceNormal*(2*dotProduct(sphereSurfaceNormal,(-eyeRay.d)))) ;

        Ray mirrorReflactanceRay = Ray(pointOnTheSphere, mirrorReflactanceRayDirection );


        bool sphereIntersection = false;
        bool triangleIntersection = false;
        bool faceIntersection = false;


        Vec3f surfaceNormal; // "intersection" function will assign this variable 


        Vec3f sphereShade   = sphereShading(scene, eyeRay, t1,  lightPosition, lightIntensity,  spheres, image, sphereIntersection, index);


        Vec3f triangleShade = triangleShading(scene, eyeRay, t2,  lightPosition, lightIntensity,  scene.triangles, image, sphereIntersection, triangleIntersection, index, surfaceNormal);
        

        Vec3f faceShade   =  faceShading(scene, eyeRay, t3,  lightPosition, lightIntensity,  scene.meshes, image, sphereIntersection, triangleIntersection, faceIntersection, index, surfaceNormal);


    }


    */
}


Vec3f ambientShader(parser::Scene& scene, parser::Material& material){
    //////////////////////////////////// AMBIENT SHADING

    float ambientRadienceRed   = scene.ambient_light.x;
    float ambientRadienceGreen = scene.ambient_light.y;
    float ambientRadienceBlue  = scene.ambient_light.z;


    Vec3f ambientShadingParams = material.ambient; // for RGB values -> between 0 and 1


    float ambientShadingRed   = ambientShadingParams.x * ambientRadienceRed; 
    float ambientShadingGreen = ambientShadingParams.y * ambientRadienceGreen; 
    float ambientShadingBlue  = ambientShadingParams.z * ambientRadienceBlue; 

    Vec3f ambientShading = Vec3f(ambientShadingRed,ambientShadingGreen,ambientShadingBlue);

    return ambientShading;
    //////////////////////////////////// AMBIENT SHADING
}


Vec3f diffuseShader(parser::Scene& scene, Ray& eyeRay, float& t, parser::Material& material , Vec3f& intersectionSurfaceNormal, Vec3f& pointOnTheMesh, Vec3f& vectorToLight, float& lightDistance, Vec3f& lightIntensity, Vec3f& irradiance){


    /*

    if(isUnderShadow(pointOnTheMesh, vectorToLight, scene, t , lightDistance, intersectionSurfaceNormal)){

        //image[index++] = 0;
        //image[index++] = 0;
        //image[index++] = 0;


        //faceShade = Vec3f(0,0,0);

        return Vec3f(0,0,0);
    }
    */

    float cosTheta = dotProduct(vectorToLight.normalize(), intersectionSurfaceNormal.normalize());

    //printf("COSTHETA: %lf \n", cosTheta );


    cosTheta = (cosTheta < 0) ? 0 : cosTheta;


    Vec3f diffuseShadingParams = material.diffuse; // for RGB values -> between 0 and 1


    //printf("Diffuse parameters: %lf , %lf , %lf \n", diffuseShadingParams.x, diffuseShadingParams.y, diffuseShadingParams.z );

    irradiance = lightIntensity * (1.0/(lightDistance*lightDistance));


    float diffuseShadingRed   = diffuseShadingParams.x * cosTheta * irradiance.x; 
    float diffuseShadingGreen = diffuseShadingParams.y * cosTheta * irradiance.y; 
    float diffuseShadingBlue  = diffuseShadingParams.z * cosTheta * irradiance.z; 

    return Vec3f(diffuseShadingRed,diffuseShadingGreen,diffuseShadingBlue);



}


Vec3f specularShader(Ray& eyeRay, Vec3f vectorToLight, Vec3f intersectionSurfaceNormal, parser::Material& material, Vec3f irradiance ){

    Vec3f halfWayVector = ((-eyeRay.d).normalize() + vectorToLight.normalize()).normalize();

    float cosAlpha = dotProduct(halfWayVector.normalize(), intersectionSurfaceNormal.normalize()); // for specular shading

    cosAlpha = (cosAlpha < 0) ? 0 : cosAlpha;


    Vec3f specularShadingParams = material.specular; // for RGB values -> between 0 and 1
    float phong_exponent = material.phong_exponent; // for RGB values -> between 0 and 1
    float cosAlphaWithPhong = pow(cosAlpha,phong_exponent); 
    //printf("Specular : %lf %lf %lf  \n", specularShadingParams.x, specularShadingParams.y, specularShadingParams.z   );


    float specularShadingRed   = specularShadingParams.x * cosAlphaWithPhong * irradiance.x; 
    float specularShadingGreen = specularShadingParams.y * cosAlphaWithPhong * irradiance.y; 
    float specularShadingBlue  = specularShadingParams.z * cosAlphaWithPhong * irradiance.z; 

    return Vec3f(specularShadingRed,specularShadingGreen,specularShadingBlue);



}


Vec3f triangleShading(parser::Scene& scene, Ray& eyeRay, float& t, parser::Face& face, parser::Material& material , Vec3f& intersectionSurfaceNormal){
    
    
    //Vec3f surfaceNormal;
    Vec3f lightPosition  = scene.point_lights[0].position; // for testing 
    Vec3f lightIntensity = scene.point_lights[0].intensity; // for testing 



    

    Vec3f irradiance;
    Vec3f pointOnTheMesh    = eyeRay.e + eyeRay.d*t; 

    Vec3f vectorToLight = -(lightPosition - pointOnTheMesh) ;

    float lightDistance = sqrt(dotProduct(vectorToLight,vectorToLight));


    if(isUnderShadow(pointOnTheMesh, vectorToLight, scene, t , lightDistance, intersectionSurfaceNormal)){

        //image[index++] = 0;
        //image[index++] = 0;
        //image[index++] = 0;


        //faceShade = Vec3f(0,0,0);

        return Vec3f(0,0,0);
    }

    Vec3f ambientShading = ambientShader(scene,  material);
    Vec3f diffuseShading = diffuseShader(scene,  eyeRay, t , material , intersectionSurfaceNormal, pointOnTheMesh, vectorToLight, lightDistance ,  lightIntensity, irradiance);
    Vec3f specularShading = specularShader(eyeRay, vectorToLight, intersectionSurfaceNormal, material, irradiance);


    Vec3f triangleShade = clamp(ambientShading + diffuseShading + specularShading);


    return triangleShade;   
}

Vec3f faceShading(parser::Scene& scene, Ray& eyeRay, float& t, parser::Face& face, parser::Material& material , Vec3f& intersectionSurfaceNormal){
    
    //Vec3f surfaceNormal;
    Vec3f lightPosition  = scene.point_lights[0].position; // for testing 
    Vec3f lightIntensity = scene.point_lights[0].intensity; // for testing 



    

    Vec3f irradiance;
    Vec3f pointOnTheMesh    = eyeRay.e + eyeRay.d*t; 

    Vec3f vectorToLight = (lightPosition - pointOnTheMesh) ;

    float lightDistance = sqrt(dotProduct(vectorToLight,vectorToLight));


    Vec3f ambientShading = ambientShader(scene,  material);

    intersectionSurfaceNormal = intersectionSurfaceNormal.normalize();


    Vec3f epsilonMovedPointOnTheMesh = pointOnTheMesh + (intersectionSurfaceNormal * scene.shadow_ray_epsilon );

    //vectorToLight = lightPosition - epsilonMovedPointOnTheMesh;

    if(isUnderShadow(epsilonMovedPointOnTheMesh, vectorToLight, scene, t , lightDistance, intersectionSurfaceNormal)){

        //image[index++] = 0;
        //image[index++] = 0;
        //image[index++] = 0;


        //faceShade = Vec3f(0,0,0);

        return clamp(ambientShading);
    }


    Vec3f diffuseShading = diffuseShader(scene,  eyeRay, t , material , intersectionSurfaceNormal, pointOnTheMesh, vectorToLight, lightDistance ,  lightIntensity, irradiance);
    Vec3f specularShading = specularShader(eyeRay, vectorToLight, intersectionSurfaceNormal, material, irradiance);


    Vec3f faceShade = clamp(ambientShading + diffuseShading + specularShading);
    //Vec3f faceShade = clamp(ambientShading+ diffuseShading );


    return faceShade;             

}




Vec3f sphereShading(parser::Scene& scene, Ray& eyeRay, float& t, parser::Sphere& sphere , parser::Material& material , Vec3f& intersectionSurfaceNormal){


    //Vec3f center = scene.vertex_data[sphere.center_vertex_id-1]; // center of the sphere 
    
    Vec3f lightPosition  = scene.point_lights[0].position; // for testing 
    Vec3f lightIntensity = scene.point_lights[0].intensity; // for testing 

    Vec3f irradiance;
    Vec3f pointOnTheSphere  = eyeRay.e + eyeRay.d*t; 

    Vec3f vectorToLight = lightPosition - pointOnTheSphere ; 

    float lightDistance = sqrt(dotProduct(vectorToLight,vectorToLight));

    Vec3f ambientShading = ambientShader(scene,  material);

    intersectionSurfaceNormal = intersectionSurfaceNormal.normalize();

    Vec3f epsilonMovedPointOnTheSphere = pointOnTheSphere + (intersectionSurfaceNormal * scene.shadow_ray_epsilon );

    //vectorToLight = lightPosition - epsilonMovedPointOnTheSphere;

    if(isUnderShadow(epsilonMovedPointOnTheSphere, vectorToLight, scene, t , lightDistance, intersectionSurfaceNormal)){

        //image[index++] = 0;
        //image[index++] = 0;
        //image[index++] = 0;


        //faceShade = Vec3f(0,0,0);

        return clamp(ambientShading);
    }

    Vec3f diffuseShading = diffuseShader(scene,  eyeRay, t , material , intersectionSurfaceNormal, pointOnTheSphere, vectorToLight, lightDistance ,  lightIntensity, irradiance);
    Vec3f specularShading = specularShader(eyeRay, vectorToLight, intersectionSurfaceNormal, material, irradiance);


    //////////////////////////////////// MIRROR SHADING
    Vec3f mirrorShadingParams = material.mirror; // for RGB values -> between 0 and 1
    //////////////////////////////////// MIRROR SHADING

    Vec3f sphereShade = clamp(ambientShading + diffuseShading + specularShading);


    //image[index++] = diffuseAndSpecular.x;
    //image[index++] = diffuseAndSpecular.y;
    //image[index++] = diffuseAndSpecular.z;
    //sphereIntersection = true;

    return sphereShade;
  
}


float minFloat(float& t1, float& t2, float& t3){

    if (t1<=t2 && t1 <= t3)
    {
        return t1;
    }else if(t2<=t1 && t2 <= t3){
        
        return t2;
    }else{
        return t3;
    }




}

Vec3f intersectionDetector(parser::Scene& scene, Ray& eyeRay, float& t_final, Vec3f& surfaceNormal, char& objInfo_0, int& objInfo_1, int& objInfo_2 ){


    float t_min = 100000.0; // We assume that all the t values will be less that this number

    float t;

    Vec3f intersectionSurfaceNormal;

    //char objectInfo[] = {' ', ' ', ' '}; 
    objInfo_0 = ' ';
    objInfo_1 = 0;
    objInfo_2 = 0;


    for (int i = 0; i < scene.spheres.size(); ++i)
    {
        Vec3f center = scene.vertex_data[scene.spheres[i].center_vertex_id-1]; // center of the sphere 
        if (intersection(eyeRay, scene.spheres[i], center ,t , surfaceNormal) && t <= t_min){
            
            intersectionSurfaceNormal = surfaceNormal;
            t_min = t;
            objInfo_0 = 's';
            objInfo_1 = i;
        }
        
    }
    for (int i = 0; i < scene.triangles.size(); ++i)
    {
        if(intersection(eyeRay, scene.triangles[i].indices, scene ,t , surfaceNormal) && t <= t_min){

            intersectionSurfaceNormal = surfaceNormal;
            t_min = t;
            objInfo_0 = 't';
            objInfo_1 = i;
        }
    }
    for (int i = 0; i < scene.meshes.size(); ++i)
    {
        for (int j = 0; j < scene.meshes[i].faces.size(); ++j)
        {
            if (intersection(eyeRay, scene.meshes[i].faces[j], scene ,t , surfaceNormal) && t <= t_min)
            {

                intersectionSurfaceNormal = surfaceNormal;
                t_min = t;
                objInfo_0 = 'f';
                objInfo_1 = i;
                objInfo_2 = j;
            }
             
        }
    }


    t_final = t_min;
    surfaceNormal = intersectionSurfaceNormal;


}

Vec3f shader(unsigned char* image, parser::Scene& scene, Ray& eyeRay, float& t, char& objInfo_0, int& objInfo_1, int& objInfo_2 ,int& index, Vec3f intersectionSurfaceNormal){

    Vec3f sphereShade;
    Vec3f triangleShade;
    Vec3f faceShade;

    if (objInfo_0 == 's')
    {
        //printf("Sphere SHADE! \n");
    
        parser::Sphere sphere = scene.spheres[objInfo_1];


        parser::Material material = scene.materials[sphere.material_id-1];        


        sphereShade = sphereShading(scene, eyeRay, t, sphere, material , intersectionSurfaceNormal);
    

    }
    else if (objInfo_0 == 't')
    {
        printf("Triangle SHADE! \n");

        parser::Face face = scene.triangles[objInfo_1].indices;

        parser::Material material = scene.materials[scene.triangles[objInfo_1].material_id-1];        

        triangleShade = faceShading(scene, eyeRay, t, face, material, intersectionSurfaceNormal );
        printf(" Triangle Shade: %lf , %lf , %lf \n" , triangleShade.x, triangleShade.y, triangleShade.z);
        printf("OBJ_INFO_1: %d\n" , objInfo_1);
    }
    else if (objInfo_0 == 'f')
    {
        //printf("Face SHADE! \n");

        parser::Face face = scene.meshes[objInfo_1].faces[objInfo_2];

        parser::Material material = scene.materials[scene.meshes[objInfo_1].material_id-1];        
    
        faceShade   =  faceShading(scene, eyeRay, t, face, material, intersectionSurfaceNormal  );
        //printf(" Face Shade: %lf , %lf , %lf \n" , faceShade.x, faceShade.y, faceShade.z);
        //printf("OBJ_INFO_1: %d\n" , objInfo_1);
        //printf("OBJ_INFO_2: %d\n" , objInfo_2);
    } else{

    }





    //float min = minFloat(t1,t2,t3);

    //printf("T1: %lf , T2: %lf , T3: %lf \n", t1,t2,t3 );
    //printf("MIN: %lf  \n" , min);


    if (objInfo_0 == 's' )
    {
        //printf("Sphere hit\n");
        image[index++] = sphereShade.x;
        image[index++] = sphereShade.y;
        image[index++] = sphereShade.z;
    } else if (objInfo_0 == 't'  )
    {
        printf("Triangle hit\n");

        image[index++] = triangleShade.x;
        image[index++] = triangleShade.y;
        image[index++] = triangleShade.z;

    } else if(objInfo_0 == 'f'  ){
        //printf("face hit\n");

        image[index++] = faceShade.x;
        image[index++] = faceShade.y;
        image[index++] = faceShade.z;

    }
    else if(objInfo_0 == ' ') {
        image[index++] = scene.background_color.x;
        image[index++] = scene.background_color.y;
        image[index++] = scene.background_color.z;
        
    }






}


int main(int argc, char* argv[])
{
    // Sample usage for reading an XML scene file
    parser::Scene scene;

    scene.loadFromXml(argv[1]);
    for (int i = 0; i < scene.cameras.size(); ++i)
    {
        std::cout << scene.cameras[i].image_name << std::endl;

        const char* filename =  scene.cameras[i].image_name.c_str();



            
        int width = scene.cameras[i].image_width;
        int height = scene.cameras[i].image_height;
        const int numOfImages = scene.cameras.size();
        
        //unsigned char** images = new unsigned char* [width * height * 3][numOfImages];

        //printf("test1\n");

        unsigned char* image = new unsigned char [width * height * 3];    

        //printf("test2\n");


        Ray gazeRay = Ray(scene.cameras[i].position , scene.cameras[i].gaze); // the eye ray which is perpendicular to the image plane

        Vec3f e = scene.cameras[i].position; // camera position, the origin of the rays we trace

        Vec3f w = scene.cameras[i].gaze; // camera gaze vector in xyz coordinates
        Vec3f v = scene.cameras[i].up; // camera up vector in xyz coordinates
        Vec3f u = crossProduct(v,-w); 

        printf("u vector: %lf , %lf , %lf\n" , u.x , u.y , u.z );

        Vec3f s;
        
        float s_u,s_v;

        int n_x = scene.cameras[i].image_width;
        int n_y = scene.cameras[i].image_height;

        float distance = scene.cameras[i].near_distance; 

        float l = scene.cameras[i].near_plane.x;
        float r = scene.cameras[i].near_plane.y;
        float b = scene.cameras[i].near_plane.z;
        float t = scene.cameras[i].near_plane.w;

        printf("width: %d \n"  , n_x);
        printf("height: %d \n" , n_y);
        printf("l: %lf , r: %lf , b: %lf , t: %lf  \n", l, r, b, t  );


        // slide -> http://saksagan.ceng.metu.edu.tr/courses/ceng477/files/pdf/week_02.pdf ------- page 13/49

        //find the coordanates of the point "q" (the point at the top-left of image plane )


        Vec3f m = e + (w) * distance ;  // m is the intersection point of the gazeRay and the image plane

        Vec3f q = m + u*l + v*t; //  

        

        //find the coordanates of the point "s" (the point we look through in ray tracing)


        Ray eyeRay ;

        printf("test\n");









        int index = 0;

        Vec3f intersectionSurfaceNormal; // "intersection" function will assign this variable 


        for (int i = 0; i < n_x; ++i)
        {
            for (int j = 0; j < n_y; ++j)
            {
                s_u = (r - l)*(j + 0.5)/n_x;
                s_v = (t - b)*(i + 0.5)/n_y;


                s = q + (u * s_u) - (v * s_v);


                eyeRay = Ray(e, (s-e).normalize());


                std::vector<parser::Mesh>     meshes    = scene.meshes;
                std::vector<parser::Triangle> triangles = scene.triangles;
                std::vector<parser::Sphere>   spheres   = scene.spheres;


                float t_final ,t1,t2,t3;

                char objInfo_0;  
                int  objInfo_1;  
                int  objInfo_2;  

                bool sphereIntersection = false;
                bool triangleIntersection = false;
                bool faceIntersection = false;


                Vec3f lightPosition  = scene.point_lights[0].position; // for testing 
                Vec3f lightIntensity = scene.point_lights[0].intensity; // for testing 


                //printf("TEST123\n");

                //printf("INDEX: %d \n", index);

                intersectionDetector(scene, eyeRay, t_final, intersectionSurfaceNormal, objInfo_0, objInfo_1, objInfo_2 );

                if (t_final < 10000.0)
                {
                    //printf("T_FINAL: %lf \n", t_final);
                    //printf("OBJ_INFO: %c , %d , %d \n", objInfo_0 , objInfo_1 ,objInfo_2);
                }


                shader(image, scene, eyeRay, t_final, objInfo_0, objInfo_1, objInfo_2, index, intersectionSurfaceNormal);


            }
        }




        write_ppm(filename, image, width, height);
        
    }
    


}
