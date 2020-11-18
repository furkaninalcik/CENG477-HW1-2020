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
        //printf("Assignment! \n" );

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

    float norm() const {
        return sqrt(x*x + y*y + z*z);
    }
    
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


parser::Scene scene;



typedef struct Vector3f
{
    union 
    {
        float x;
        float r;
    };
    union
    {
        float y;
        float g;
    };
    union
    {
        float z;
        float b;
    };
} Vector3f;



class Ray2{


    private:
        

    public:
        
        Vec3f origin;
        Vec3f direction;

        Ray2(){
            //printf("empty ray constructor\n");
        }
        Ray2(Vec3f o, Vec3f d ){

            //printf("ray constructor\n");


            origin = o;
            direction = d;
        }

        Ray2(Vector3f o, Vector3f d ){

            //printf("ray constructor\n");


            origin.x = o.x;
            origin.y = o.y;
            origin.z = o.z;


            direction.x = d.x;
            direction.y = d.y;
            direction.z = d.z;
        }


        Vec3f RayVect(float t){

            Vec3f v2 = Vec3f(direction.x*t, direction.y*t, direction.z*t);

            Vec3f result = Vec3f(origin.x + v2.x, origin.y + v2.y, origin.z + v2.z );

            return result;
        }
       
        //~Ray();
    
};





typedef struct IntersectionResult
{

    
    bool intersection = false;
    Vector3f intersection_point;
    Vector3f surface_normal;
    float t;  



} IntersectionResult;







Ray2 mirrorReflectanceRay(Ray2 primaryRay, IntersectionResult intersection_info){
    
    Vec3f incomingRayDirection, surfaceNormal;
    
    incomingRayDirection = Vec3f(primaryRay.direction.x, primaryRay.direction.y, primaryRay.direction.z ).normalize();
    surfaceNormal = Vec3f(intersection_info.surface_normal.x, intersection_info.surface_normal.y, intersection_info.surface_normal.z).normalize();

    //Vec3f outGoingRayDirection  = incomingRayDirection + (surfaceNormal*(2*(surfaceNormal).dotProduct((-incomingRayDirection))));
    Vec3f outGoingRayDirection  = incomingRayDirection + (surfaceNormal*(2*dotProduct((surfaceNormal),(-incomingRayDirection))));

    Vector3f direction;
    direction.x = outGoingRayDirection.x;
    direction.y = outGoingRayDirection.y;
    direction.z = outGoingRayDirection.z;

    Ray2 outGoingRay =  Ray2(intersection_info.intersection_point,direction);


    return outGoingRay;


}



Vec3f computeLightContribution2(parser::PointLight light, Vector3f p)
{
    /***********************************************
     *                                             *
     * TODO: Implement this function               *
     *                                             *
     ***********************************************
     */



    Vec3f point_vector(p.x,p.y,p.z);

    Vec3f position_vector = light.position;

    Vec3f intensity_vector = light.intensity;

    float distance = (point_vector - position_vector).norm();

    intensity_vector = intensity_vector * (1/(distance*distance));

    return intensity_vector;

    //Vector3f intensity_result ;
    //intensity_result.x = intensity_vector.x; 
    //intensity_result.y = intensity_vector.y; 
    //intensity_result.z = intensity_vector.z; 


    //return intensity_result;

}


Vec3f diffuseShader2( int mat_id, int light_id, Ray2 ray, Vec3f surface_normal, Vec3f intersection_point){


    Vec3f light_position = scene.point_lights[light_id].position;

    Vec3f vectorToLight = light_position - intersection_point;


    //float cosTheta = (vectorToLight.normalize()).dotProduct(surface_normal.normalize());
    float cosTheta = dotProduct((vectorToLight.normalize()), surface_normal.normalize());

    //printf("COSTHETA: %lf \n", cosTheta );


    cosTheta = (cosTheta < 0) ? 0 : cosTheta;
    //std::cout << cosTheta << endl;



    Vec3f diffuseShadingParams = scene.materials[mat_id-1].diffuse; // for RGB values -> between 0 and 1

    //printf("Diffuse parameters: %lf , %lf , %lf \n", diffuseShadingParams.x, diffuseShadingParams.y, diffuseShadingParams.z );




    //irradiance = lightIntensity * (1.0/(lightDistance*lightDistance));

    Vector3f intersection_point_vector3f ;
    intersection_point_vector3f.x = intersection_point.x; 
    intersection_point_vector3f.y = intersection_point.y; 
    intersection_point_vector3f.z = intersection_point.z; 
/*
    Vec3f irradiance = Vec3f(0,0,0);

    for (int i = 0; i < lights.size(); ++i)
    {
        irradiance = irradiance + lights[i]->computeLightContribution(intersection_point_vector3f);

    }
*/
    Vec3f irradiance = computeLightContribution2(scene.point_lights[light_id], intersection_point_vector3f);

    float diffuseShadingRed   = diffuseShadingParams.x * cosTheta * irradiance.x; 
    float diffuseShadingGreen = diffuseShadingParams.y * cosTheta * irradiance.y; 
    float diffuseShadingBlue  = diffuseShadingParams.z * cosTheta * irradiance.z; 

    return Vec3f(diffuseShadingRed,diffuseShadingGreen,diffuseShadingBlue);



}

Vec3f ambientShader2( int mat_id){
    //////////////////////////////////// AMBIENT SHADING

    float ambientRadienceRed   = scene.ambient_light.x;
    float ambientRadienceGreen = scene.ambient_light.y;
    float ambientRadienceBlue  = scene.ambient_light.z;


    Vec3f ambientShadingParams = scene.materials[mat_id-1].ambient; // for RGB values -> between 0 and 1


    float ambientShadingRed   = ambientShadingParams.x * ambientRadienceRed; 
    float ambientShadingGreen = ambientShadingParams.y * ambientRadienceGreen; 
    float ambientShadingBlue  = ambientShadingParams.z * ambientRadienceBlue; 

    Vec3f ambientShading = Vec3f(ambientShadingRed,ambientShadingGreen,ambientShadingBlue);

    return ambientShading;
    //////////////////////////////////// AMBIENT SHADING
}

Vec3f specularShader2( int mat_id, int light_id, Ray2 ray, Vec3f surface_normal, Vec3f intersection_point){


    Vec3f light_position = scene.point_lights[light_id].position;

    Vec3f vectorToLight = light_position - intersection_point;

    Vec3f minus_ray_direction = ray.direction;
    minus_ray_direction = -minus_ray_direction;

    Vec3f halfWayVector = ((minus_ray_direction.normalize()) + vectorToLight.normalize()).normalize();

    //float cosAlpha = (halfWayVector.normalize()).dotProduct(surface_normal.normalize()); // for specular shading
    float cosAlpha = dotProduct((halfWayVector.normalize()), surface_normal.normalize()); // for specular shading

    cosAlpha = (cosAlpha < 0) ? 0 : cosAlpha;


    Vec3f specularShadingParams = scene.materials[mat_id-1].specular; // for RGB values -> between 0 and 1
    float phong_exponent = scene.materials[mat_id-1].phong_exponent; // for RGB values -> between 0 and 1
    float cosAlphaWithPhong = pow(cosAlpha,phong_exponent); 
    //printf("Specular : %lf %lf %lf  \n", specularShadingParams.x, specularShadingParams.y, specularShadingParams.z   );

    Vector3f intersection_point_vector3f ;
    intersection_point_vector3f.x = intersection_point.x; 
    intersection_point_vector3f.y = intersection_point.y; 
    intersection_point_vector3f.z = intersection_point.z; 

    Vec3f irradiance = computeLightContribution2(scene.point_lights[light_id], intersection_point_vector3f);


    float specularShadingRed   = specularShadingParams.x * cosAlphaWithPhong * irradiance.x; 
    float specularShadingGreen = specularShadingParams.y * cosAlphaWithPhong * irradiance.y; 
    float specularShadingBlue  = specularShadingParams.z * cosAlphaWithPhong * irradiance.z; 

    return Vec3f(specularShadingRed,specularShadingGreen,specularShadingBlue);



}




IntersectionResult intersect(parser::Sphere sphere, Ray2 ray)
{
    /***********************************************
     *                                             *
     * TODO: Implement this function               *
     *                                             *
     ***********************************************
     */

    float t ;
    IntersectionResult intersection_info;

    Vec3f e = ray.origin;
    Vec3f d = ray.direction;

    //Vec3f center = Vec3f(-0.875, 1, -2);
    
    //Vec3f center = (*vertices)[cIndex-1];
    Vec3f center = scene.vertex_data[sphere.center_vertex_id-1];

    //float r = R; // radius of the sphere
    float r = sphere.radius; // radius of the sphere


    float a = dotProduct(d, d);           // a is A in the equation -> At^2 + Bt + C = 0 // 
    float b = 2*(dotProduct(d, (e-center)));       // b is B in the equation -> At^2 + Bt + C = 0 // 
    float c = dotProduct((e-center), (e-center)) - r*r; // c is C in the equation -> At^2 + Bt + C = 0 // 

    float discriminant = b*b - 4*a*c;

    if (discriminant < 0.005) // 
    {
        intersection_info.intersection = false;
        //return false;
    }
    else{
        float x0 = (-b - sqrt(discriminant))/(2*a); // one of the real roots of the equation
        float x1 = (-b + sqrt(discriminant))/(2*a); // one of the real roots of the equation
        t = (x0 < x1) ? x0 : x1;
        //printf("t1 %lf \n", x0 );
        //printf("t2 %lf \n", x1 );
        if (t < 0)
        {   
            //std::cout << "Sphere Intersection" << t << endl;
            //std::cout << "Negative T: " << t << endl;
            
            intersection_info.intersection = false;

        } else{

            Vec3f pointOnTheSphere  = e + d*t; 

            Vector3f intersectionPoint;
            intersectionPoint.x = pointOnTheSphere.x; 
            intersectionPoint.y = pointOnTheSphere.y; 
            intersectionPoint.z = pointOnTheSphere.z;

            Vector3f surfaceNormal;
            Vec3f surfaceNormal_Temp = (pointOnTheSphere - center) * (1.0 / r);
            surfaceNormal.x = surfaceNormal_Temp.x; 
            surfaceNormal.y = surfaceNormal_Temp.y; 
            surfaceNormal.z = surfaceNormal_Temp.z;

            intersection_info.intersection_point = intersectionPoint;
            intersection_info.surface_normal = surfaceNormal;
            intersection_info.intersection = true;
            intersection_info.t = t;

            //return true;     
        }

   
    }

    return intersection_info;

    //Vec3f c = sphere.vertex_data[scene.center_vertex_id]; // center of the sphere
}






IntersectionResult intersect(parser::Triangle triangle, Ray2 ray){

    /***********************************************
     *                                             *
     * TODO: Implement this function               *
     *                                             *
     ***********************************************
     */

    IntersectionResult intersection_info;

    Vec3f e = ray.origin; // origin 
    Vec3f d = ray.direction; // direction

    Vec3f p ; // the ray-plane intersection point (may or may not be inside the triangle) 

    float gama, beta; // variables for barycentric coordinates

    float t;


    //Vec3f v1 = (*vertices)[p1Index-1];
    //Vec3f v2 = (*vertices)[p2Index-1];
    //Vec3f v3 = (*vertices)[p3Index-1];

    Vec3f v1 = scene.vertex_data[triangle.indices.v0_id-1];
    Vec3f v2 = scene.vertex_data[triangle.indices.v1_id-1];
    Vec3f v3 = scene.vertex_data[triangle.indices.v2_id-1];



    // calculating plane normal

    Vec3f vector_for_cross_product;


    //Vec3f normalVector = vector_for_cross_product.crossProduct( v3-v2 , v2-v1);  // BE CAREFULL ABOUT THE ORDER OF THE VERTICES
    Vec3f normalVector = crossProduct( v3-v2 , v2-v1);  // BE CAREFULL ABOUT THE ORDER OF THE VERTICES



    Vec3f surfaceNormal_Temp = -normalVector; // TO BE USED BY SHADING PART OF THE CODE

    if (dotProduct(normalVector,d)  < 0.000001) // if plane and ray are parallel 
    {
        intersection_info.intersection = false;
        return intersection_info;
        //return false;
    }

    t = (dotProduct((v1 - e),normalVector))/(dotProduct(d, normalVector)); // calculating t to find the ray-plane intersection point "p"


    //printf("t : %lf \n" , t);

    p = e + d * t;



    if (t < 0)
    {   
        //std::cout << "Triangle Intersection" << t << endl;
        //std::cout << "Negative T: " << t << endl;
        intersection_info.intersection = false;
        return intersection_info;
    }




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
        intersection_info.intersection = false;
        return intersection_info;
        //return false;
    }

    //printf("TEST3\n");


    beta = (p1*p5 - p2*p4) / den; // BETA OR GAMA ???

    if (beta < 0 || beta > 1-gama)
    {
        intersection_info.intersection = false;
        return intersection_info;

        //return false;
    }

    //printf("TEST4\n");
    else{

        Vec3f pointOnTheTriangle  = p; 

        Vector3f intersectionPoint;
        intersectionPoint.x = pointOnTheTriangle.x; 
        intersectionPoint.y = pointOnTheTriangle.y; 
        intersectionPoint.z = pointOnTheTriangle.z;

        Vector3f surfaceNormal;

        surfaceNormal.x = surfaceNormal_Temp.x; 
        surfaceNormal.y = surfaceNormal_Temp.y; 
        surfaceNormal.z = surfaceNormal_Temp.z;

        intersection_info.intersection_point = intersectionPoint;
        intersection_info.surface_normal = surfaceNormal;
        intersection_info.intersection = true;
        intersection_info.t = t;

        return intersection_info;

    //return true;

    }
}




IntersectionResult intersect(parser::Mesh mesh, Ray2 ray)
{
    /***********************************************
     *                                             *
     * TODO: Implement this function               *
     *                                             *
     ***********************************************
     */

    int t_min = 100000.0;

    int num_of_faces = mesh.faces.size();

    IntersectionResult intersection_info_array[num_of_faces];

    IntersectionResult first_intersection_info ;
    
    IntersectionResult face_intersection_info ;
    IntersectionResult closest_intersection_info ;

    //std::cout << faces.size() << endl;


    for (int i = 0; i < mesh.faces.size(); ++i)
    {    
        //intersection_info_array[i] = faces[i].intersect(ray); //assigning the list of intersection_infos of the faces
     

        //face_intersection_info = mesh.faces[i].intersect(ray);

        
        parser::Triangle tr {
            mesh.material_id,
            mesh.faces[i]
        };


        face_intersection_info = intersect(tr, ray);

        if (face_intersection_info.intersection && face_intersection_info.t <= t_min)
        {

            t_min = face_intersection_info.t;
            closest_intersection_info = face_intersection_info;
            
        }


    }


    //return intersection_info_array[0]; // returning the first element of the array that is the info about the first face of the mesh
    return closest_intersection_info; // returning the intersection_info of the first intersection without checking if it is the closest face to the camera 


}




Vec3f shade(Ray2 primaryRay, int recursionTracker){

    recursionTracker -= 1;

    IntersectionResult intersection_info;
    IntersectionResult closest_intersection_info;
    closest_intersection_info.t = 10000000;
    closest_intersection_info.intersection = false;

    intersection_info.intersection=false;


    int closest_material_id; // material id of the closest object
    int mat_id ;
    
    Vec3f diffuse;
    Vec3f specular;
    Vec3f mirror;
    Vec3f ambient;

    for (int k = 0; k < scene.spheres.size(); ++k)
    {

        
        intersection_info = intersect(scene.spheres[k], primaryRay);
        mat_id = scene.spheres[k].material_id;

        if (intersection_info.intersection && intersection_info.t < closest_intersection_info.t)
        {
            closest_intersection_info = intersection_info;
            closest_material_id = mat_id;
        }

    }

    for (int k = 0; k < scene.triangles.size(); ++k)
    {

        
        intersection_info = intersect(scene.triangles[k], primaryRay);
        mat_id = scene.triangles[k].material_id;

        if (intersection_info.intersection && intersection_info.t < closest_intersection_info.t)
        {
            closest_intersection_info = intersection_info;
            closest_material_id = mat_id;
        }

    }

    for (int k = 0; k < scene.meshes.size(); ++k)
    {

        
        intersection_info = intersect(scene.meshes[k], primaryRay);
        mat_id = scene.meshes[k].material_id;

        if (intersection_info.intersection && intersection_info.t < closest_intersection_info.t)
        {
            closest_intersection_info = intersection_info;
            closest_material_id = mat_id;
        }

    }

    if (!closest_intersection_info.intersection)
    {
        return {scene.background_color.x, scene.background_color.y, scene.background_color.z};
        //return {10, 10, 100};
    }

    else{
        
        ambient  =  ambientShader2(closest_material_id);

        diffuse  = Vec3f(0,0,0);
        specular = Vec3f(0,0,0);
        mirror   = Vec3f(0,0,0);

        for (int light_id = 0; light_id < scene.point_lights.size(); ++light_id)
        {

            bool shadowRay_object_intersection = false;
            //float epsilon =  shadowRayEps;
            float epsilon =  scene.shadow_ray_epsilon;
            //epsilon =  1000000;


            Vec3f light_position = scene.point_lights[light_id].position;

            Vec3f intPoint(closest_intersection_info.intersection_point.x, closest_intersection_info.intersection_point.y, closest_intersection_info.intersection_point.z);

            


            Vec3f intersection_point_to_light = (light_position - intPoint).normalize();

            Vector3f shadowRay_origin, shadowRay_direction;

            shadowRay_origin.x = closest_intersection_info.intersection_point.x + intersection_point_to_light.x * epsilon;
            shadowRay_origin.y = closest_intersection_info.intersection_point.y + intersection_point_to_light.y * epsilon;
            shadowRay_origin.z = closest_intersection_info.intersection_point.z + intersection_point_to_light.z * epsilon;

            shadowRay_direction.x = intersection_point_to_light.x;
            shadowRay_direction.y = intersection_point_to_light.y;
            shadowRay_direction.z = intersection_point_to_light.z;

            
            Ray2 shadowRay = Ray2(shadowRay_origin, shadowRay_direction );


            float t_to_object;

            IntersectionResult shadowRay_intersection_info;

            

            for (int k = 0; k < scene.spheres.size(); ++k)
            {

                
                shadowRay_intersection_info = intersect(scene.spheres[k], shadowRay);

                if (shadowRay_intersection_info.intersection){

                    //Vec3f intersection_point = closest_intersection_info.intersection_point;
                    Vec3f intersection_point(closest_intersection_info.intersection_point.x, closest_intersection_info.intersection_point.y, closest_intersection_info.intersection_point.z);

                    //Vec3f shadowRay_intersection_point = shadowRay_intersection_info.intersection_point;
                    Vec3f shadowRay_intersection_point(shadowRay_intersection_info.intersection_point.x,shadowRay_intersection_info.intersection_point.y,shadowRay_intersection_info.intersection_point.z);

                    float light_intersectionPoint_distance    = (light_position - intersection_point).norm();
                    float intersectionPoint_obstacle_distance = (shadowRay_intersection_point - intersection_point).norm();

                    if (light_intersectionPoint_distance > intersectionPoint_obstacle_distance)
                    {
                        shadowRay_object_intersection = true;
                        break;
                    }


                }  

            }

            if (!shadowRay_object_intersection)
            {
                for (int k = 0; k < scene.triangles.size(); ++k)
                {
                    
                    shadowRay_intersection_info = intersect(scene.triangles[k], shadowRay);

                    if (shadowRay_intersection_info.intersection){

                        //Vec3f intersection_point = closest_intersection_info.intersection_point;
                        Vec3f intersection_point(closest_intersection_info.intersection_point.x, closest_intersection_info.intersection_point.y, closest_intersection_info.intersection_point.z);

                        //Vec3f shadowRay_intersection_point = shadowRay_intersection_info.intersection_point;
                        Vec3f shadowRay_intersection_point(shadowRay_intersection_info.intersection_point.x,shadowRay_intersection_info.intersection_point.y,shadowRay_intersection_info.intersection_point.z);

                        float light_intersectionPoint_distance    = (light_position - intersection_point).norm();
                        float intersectionPoint_obstacle_distance = (shadowRay_intersection_point - intersection_point).norm();

                        if (light_intersectionPoint_distance > intersectionPoint_obstacle_distance)
                        {
                            shadowRay_object_intersection = true;
                            break;
                        }


                    }  

                }
                
            }

            if (!shadowRay_object_intersection)
            {
                for (int k = 0; k < scene.meshes.size(); ++k)
                {
                    
                    shadowRay_intersection_info = intersect(scene.meshes[k], shadowRay);

                    if (shadowRay_intersection_info.intersection){

                        //Vec3f intersection_point = closest_intersection_info.intersection_point;
                        Vec3f intersection_point(closest_intersection_info.intersection_point.x, closest_intersection_info.intersection_point.y, closest_intersection_info.intersection_point.z);

                        //Vec3f shadowRay_intersection_point = shadowRay_intersection_info.intersection_point;
                        Vec3f shadowRay_intersection_point(shadowRay_intersection_info.intersection_point.x,shadowRay_intersection_info.intersection_point.y,shadowRay_intersection_info.intersection_point.z);

                        float light_intersectionPoint_distance    = (light_position - intersection_point).norm();
                        float intersectionPoint_obstacle_distance = (shadowRay_intersection_point - intersection_point).norm();

                        if (light_intersectionPoint_distance > intersectionPoint_obstacle_distance)
                        {
                            shadowRay_object_intersection = true;
                            break;
                        }


                    }  

                }
                
            }





            if (!shadowRay_object_intersection)
            {
                Vec3f sn(closest_intersection_info.surface_normal.x, closest_intersection_info.surface_normal.y, closest_intersection_info.surface_normal.z);
                Vec3f ip(closest_intersection_info.intersection_point.x, closest_intersection_info.intersection_point.y, closest_intersection_info.intersection_point.z);

                diffuse  = diffuse  + diffuseShader2( closest_material_id, light_id, primaryRay, sn, ip);
                specular = specular + specularShader2( closest_material_id, light_id, primaryRay, sn, ip);

                //diffuse  = diffuse  + diffuseShader2(scene, mat_id, light_id, primaryRay, intersection_info.surface_normal, intersection_info.intersection_point);
                //specular = specular + specularShader2(scene, mat_id, light_id, primaryRay, intersection_info.surface_normal, intersection_info.intersection_point);

            }



        }



        
        
        if (recursionTracker > 0)
        {
            Ray2 ReflectanceRay = mirrorReflectanceRay(primaryRay,closest_intersection_info );
            mirror = shade( ReflectanceRay, recursionTracker );

            
            
            Vec3f mirrorShadingParams;

            //mirrorShadingParams = scene.materials[closest_material_id-1].mirror;
            //mirrorShadingParams = scene.materials[0].mirror;
            

            mirrorShadingParams.x = (float)scene.materials[closest_material_id-1].mirror.x;
            mirrorShadingParams.y = (float)scene.materials[closest_material_id-1].mirror.y;
            mirrorShadingParams.z = (float)scene.materials[closest_material_id-1].mirror.z;

            //mirrorShadingParams = scene.materials[objects[k].material_id-1].mirror; // for RGB values -> between 0 and 1

            //mirror = clamp(mirror);

            //float mirrorShadingRed   = mirrorShadingParams.x * mirror.x ; 
            //float mirrorShadingGreen = mirrorShadingParams.y * mirror.y ; 
            //float mirrorShadingBlue  = mirrorShadingParams.z * mirror.z ; 

            //mirror = Vec3f(mirrorShadingRed, mirrorShadingGreen, mirrorShadingBlue);
            mirror = Vec3f(mirrorShadingParams.x * mirror.x, mirrorShadingParams.y * mirror.y, mirrorShadingParams.z * mirror.z);

            //mirror = Vec3f(0, 0, 0);


        }



        //Vec3f clamp_vector; 
        //Vec3f clamped_shade = clamp_vector.clamp(ambient + diffuse + specular + mirror);
        Vec3f clamped_shade = clamp(ambient + diffuse + specular + mirror);

        return {clamped_shade.x,clamped_shade.y,clamped_shade.z};

        //img->setPixelValue(j,i,{clamped_shade.x,clamped_shade.y,clamped_shade.z});
        //img->setPixelValue(j,i,{200,200,200});
        


        //break;
    }


    
    


}







Ray2 getPrimaryRay(parser::Camera camera, int col, int row)
{
    /***********************************************
     *                                             *
     * TODO: Implement this function               *
     *                                             *
     ***********************************************
     */


    //Vec3f position;
    //Vec3f gaze;
    //Vec3f up;
    //Vec4f near_plane;
    //float near_distance;
    //int image_width, image_height;
    //std::string image_name;


     Ray2 gazeRay = Ray2(camera.position, camera.gaze); // the eye ray which is perpendicular to the image plane

     Vec3f e = camera.position; // camera position, the origin of the rays we trace

     Vec3f w = camera.gaze; // camera gaze vector in xyz coordinates
     Vec3f v = camera.up; // camera up vector in xyz coordinates

     Vec3f vector_for_member_function ; // created to use crossProduct;

     //Vec3f u = vector_for_member_function.crossProduct(v,-w); 
     Vec3f u = crossProduct(v,-w); 


     Vec3f s;

     float s_u,s_v;



     int n_x = camera.image_width;
     int n_y = camera.image_height;

     float distance = camera.near_distance; 

     float l = camera.near_plane.x;
     float r = camera.near_plane.y;
     float b = camera.near_plane.z;
     float t = camera.near_plane.w;
     /*
     int n_x = imgPlane.nx;
     int n_y = imgPlane.ny;

     float distance = imgPlane.distance; 

     float l = imgPlane.left;
     float r = imgPlane.right;
     float b = imgPlane.bottom;
     float t = imgPlane.top;
    */



     // slide -> http://saksagan.ceng.metu.edu.tr/courses/ceng477/files/pdf/week_02.pdf ------- page 13/49




     Vec3f m = e + (w) * distance ;  // m is the intersection point of the gazeRay and the image plane

     Vec3f q = m + u*l + v*t; // find the coordanates of the point "q" (the point at the top-left of image plane )


     Ray2 eyeRay ;

     s_u = (r - l)*(row + 0.5)/n_x;
     s_v = (t - b)*(col + 0.5)/n_y;


     s = q + (u * s_u) - (v * s_v);


     Vector3f origin_point, direction_vector;
     
     origin_point.x = e.x;
     origin_point.y = e.y;
     origin_point.z = e.z;

     Vec3f normalized_direction_vector = (s-e).normalize();

     direction_vector.x = normalized_direction_vector.x;
     direction_vector.y = normalized_direction_vector.y;
     direction_vector.z = normalized_direction_vector.z;

     eyeRay = Ray2(origin_point, direction_vector);

     return eyeRay;


}



// aşağıyı sil 
/*
typedef union Color
{
    struct
    {
        unsigned char red;
        unsigned char grn;
        unsigned char blu;
    };

    unsigned char channel[3];
} Color;



class Image
{
    public:
    Color** data;                   // Image data
    int width;                      // Image width
    int height;                     // Image height

    Image(int width, int height);   // Constructor
    void setPixelValue(int col, int row, const Color& color); // Sets the value of the pixel at the given column and row
    void saveImage(const char *imageName) const;              // Takes the image name as a file and saves it as a ppm file. 
};


Image::Image(int width, int height)
    : width(width), height(height)
    {
        data = new Color* [height];

        for (int y = 0; y < height; ++y)
        {
            data[y] = new Color [width];
        }
    }

//
// Set the value of the pixel at the given column and row
//
void Image::setPixelValue(int col, int row, const Color& color)
{
    data[row][col] = color;
}

void Image::saveImage(const char *imageName) const
{
    FILE *output;

    output = fopen(imageName, "w");
    fprintf(output, "P3\n");
    fprintf(output, "%d %d\n", width, height);
    fprintf(output, "255\n");

    for(int y = 0 ; y < height; y++)
    {
        for(int x = 0 ; x < width; x++)
        {
            for (int c = 0; c < 3; ++c)
            {
                fprintf(output, "%d ", data[y][x].channel[c]);
            }
        }

        fprintf(output, "\n");
    }

    fclose(output);
}
*/

int main(int argc, char* argv[])
{
    // Sample usage for reading an XML scene file
    //parser::Scene scene;

    scene.loadFromXml(argv[1]);



    //printf("Mirror Reflectance Parameters: %d %d %d\n", scene.materials[0].mirror.x, scene.materials[0].mirror.y, scene.materials[0].mirror.z);
    //printf("Diffues Reflectance Parameters: %lf %lf %lf\n", scene.materials[0].diffuse.x, scene.materials[0].diffuse.y, scene.materials[0].diffuse.z);
    //printf("Diffues Reflectance Parameters: %lf %lf %lf\n", scene.materials[1].diffuse.x, scene.materials[1].diffuse.y, scene.materials[1].diffuse.z);



    const char* filename;
    unsigned char* image;

    Ray2 primaryRay ;
    Vec3f final_shade ;

    //Image* img;

    int recursionTracker = scene.max_recursion_depth;
    
    for (int camera_id = 0; camera_id < scene.cameras.size(); ++camera_id)
    {
        filename =  scene.cameras[camera_id].image_name.c_str();
        int width = scene.cameras[camera_id].image_width;
        int height = scene.cameras[camera_id].image_height;

        image = new unsigned char [width * height * 3];    

        //img = new Image(width,height);


        //recursionTracker = 2;

        //std::cout << width << endl;
        //std::cout << height << endl; 

        int index = 0;
        for (int i = 0; i < height; ++i)
        {
            for (int j = 0; j < width; ++j)
            {
                


                //Ray2 primaryRay = getPrimaryRay(scene.cameras[camera_id],j,i);
                primaryRay = getPrimaryRay(scene.cameras[camera_id],i,j);

                final_shade = shade(primaryRay, recursionTracker);

                printf("Pixel %d %d is colored\n", i,j);

                //Color shade_result;
                //shade_result.red = final_shade.x;
                //shade_result.grn = final_shade.y;
                //shade_result.blu = final_shade.z;

                image[index++] = final_shade.x;
                image[index++] = final_shade.y;
                image[index++] = final_shade.z;


                //img->setPixelValue(i,j,shade_result);
                //setPixelValue(i,j,shade_result);
                
                    
            }
        }




        //img->saveImage(filename);

        write_ppm(filename, image, scene.cameras[camera_id].image_width, scene.cameras[camera_id].image_height);



    }


    


}
