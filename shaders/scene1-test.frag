#version 410

uniform vec2 u_resolution; // viewport size in pixels (width, height)
uniform int u_frame; // frame increment counter
uniform vec2 u_mouse;
uniform vec2 u_cameraRot;
uniform vec3 u_cameraPos;
uniform float u_time;
uniform vec3 u_tempColor;
uniform float u_focusDist;
uniform float u_aperture;

uniform sampler2D u_pass1;

uniform sampler2D u_ground;
uniform sampler2D u_ground_normal;
uniform sampler2D u_ground_disp;
uniform sampler2D u_ground_roughness;

uniform sampler2D u_onyx;
uniform sampler2D u_onyx_roughness;
uniform sampler2D u_onyx_displacement;

uniform sampler2D u_tile;
uniform sampler2D u_tile_roughness;
uniform sampler2D u_tile_displacement;

uniform sampler2D u_main;
uniform int u_spf; // 16, [1, 64]
out vec4 fragColor;

// a pixel value multiplier of light before tone mapping and sRGB
const float c_exposure = 0.5f;
const float KEY_SPACE = 32.5 / 256.0;
const float c_rayPosNormalNudge = 0.01f;

vec3 LessThan(vec3 f, float value) {
  return vec3((f.x < value) ? 1.0f : 0.0f, (f.y < value) ? 1.0f : 0.0f,
              (f.z < value) ? 1.0f : 0.0f);
}

vec3 LinearToSRGB(vec3 rgb) {
  rgb = clamp(rgb, 0.0f, 1.0f);
  return mix(pow(rgb, vec3(1.0f / 2.4f)) * 1.055f - 0.055f, rgb * 12.92f,
             LessThan(rgb, 0.0031308f));
}

vec3 SRGBToLinear(vec3 rgb) {
  rgb = clamp(rgb, 0.0f, 1.0f);
  return mix(pow(((rgb + 0.055f) / 1.055f), vec3(2.4f)), rgb / 12.92f,
             LessThan(rgb, 0.04045f));
}

vec3 ACESFilm(vec3 x) {
  float a = 2.51f;
  float b = 0.03f;
  float c = 2.43f;
  float d = 0.59f;
  float e = 0.14f;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0f, 1.0f);
}

float FresnelReflectAmount(float n1, float n2, vec3 normal, vec3 incident,
                           float f0, float f90) {
  float r0 = (n1 - n2) / (n1 + n2);
  r0 *= r0;
  float cosX = -dot(normal, incident);
  if (n1 > n2) {
    float n = n1 / n2;
    float sinT2 = n * n * (1.0 - cosX * cosX);
    if (sinT2 > 1.0)
      return f90;
    cosX = sqrt(1.0 - sinT2);
  }
  float x = 1.0 - cosX;
  float ret = r0 + (1.0 - r0) * x * x * x * x * x;
  return mix(f0, f90, ret);
}

const int MAX_STEPS = 192;
const float EPSILON = 0.001;
const float MAX_DIST = 500.0;
const float c_FOVDegrees = 90.0f;
const float c_pi = 3.14159265359f;
const float c_twopi = 2.0f * c_pi;
const float c_superFar = 10000.0f;
const int c_numBounces = 8;
const int SHADOW_MAX_STEPS = 96;

// --- NEW RECTANGULAR LIGHT CONSTANTS ---
const vec3 c_lightPos = vec3(0.0, 0.5, 0.5);
const vec2 c_lightSize = vec2(0.2, 0.2); // X and Z half-extents
const float c_lightEmissionStrength = 20.0;

struct Ray {
  vec3 origin;
  vec3 direction;
  vec3 target;
};

struct Material {
  vec3 albedo;     
  float roughness; 
  float metallic;
  vec3 emissive;
  float transmission;
  float IOR;
  vec3 absorption;
};

struct SDF {
  float distance;
  float id;
};

struct SRayHitInfo {
  bool fromInside;
  float dist;
  vec3 normal;
  Material material;
};

float opUnion(float a, float b) { return min(a, b); }

SDF opUnionID(SDF res1, SDF res2) {
  if (res1.distance < res2.distance) {
    return res1;
  } else {
    return res2;
  }
}

// PCG random
uint NextRandom(inout uint state) {
  state = state * 747796405u + 2891336453u;
  uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  uint result = (word >> 22u) ^ word;
  return result;
}

float RandomValue(inout uint state) {
  return float(NextRandom(state)) / 4294967295.0; // 2^32 - 1
}

float RandomValueNormalDistribution(inout uint state) {
  float u1 = RandomValue(state);
  float u2 = RandomValue(state);
  u1 = max(u1, 1e-7);
  float theta = 2.0 * 3.14159265359 * u2;
  float rho = sqrt(-2.0 * log(u1));
  return rho * cos(theta);
}

vec3 RandomDirection(inout uint state) {
  float x = RandomValueNormalDistribution(state);
  float y = RandomValueNormalDistribution(state);
  float z = RandomValueNormalDistribution(state);
  return normalize(vec3(x, y, z));
}

vec3 RandomUnitVector(inout uint state) {
  float z = RandomValue(state) * 2.0f - 1.0f;
  float a = RandomValue(state) * c_twopi;
  float r = sqrt(1.0f - z * z);
  float x = r * cos(a);
  float y = r * sin(a);
  return vec3(x, y, z);
}

vec2 RandomPointInCircle(inout uint state) {
  float angle = RandomValue(state) * c_twopi;
  float radius = sqrt(RandomValue(state)); 
  return vec2(cos(angle), sin(angle)) * radius;
}

vec3 ImportanceSampleGGX(vec2 Xi, vec3 N, float roughness) {
  float a = roughness * roughness;
  float phi = 2.0 * c_pi * Xi.x;
  float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a * a - 1.0) * Xi.y));
  float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

  vec3 H;
  H.x = cos(phi) * sinTheta;
  H.y = sin(phi) * sinTheta;
  H.z = cosTheta;

  vec3 up = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
  vec3 tangent = normalize(cross(up, N));
  vec3 bitangent = cross(N, tangent);

  vec3 sampleVec = tangent * H.x + bitangent * H.y + N * H.z;
  return normalize(sampleVec);
}

float dot2(vec3 v) { return dot(v, v); }

float sdSphere(vec3 p, float r) { return length(p) - r; }

float sdPlane(vec3 p, vec3 n, float h) { return dot(p, n) + h; }

float sdBox(vec3 p, vec3 b) {
  vec3 q = abs(p) - b;
  return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

void pR(inout vec2 p, float a) {
  p = cos(a) * p + sin(a) * vec2(p.y, -p.x);
}

Material GetZeroedMaterial() {
  Material ret;
  ret.albedo = vec3(0.0);
  ret.roughness = 1.0;
  ret.metallic = 0.0;
  ret.emissive = vec3(0.0);
  ret.transmission = 0.0;
  ret.IOR = 1.0;
  ret.absorption = vec3(0.0);
  return ret;
}

vec3 triPlanar(sampler2D tex, vec3 p, vec3 normal) {
  normal = abs(normal);
  normal = pow(normal, vec3(5.0));
  normal /= normal.x + normal.y + normal.z;
  return (texture(tex, p.xy * 0.5 + 0.5) * normal.z +
          texture(tex, p.xz * 0.5 + 0.5) * normal.y +
          texture(tex, p.yz * 0.5 + 0.5) * normal.x).rgb;
}

vec3 triPlanarNormal(sampler2D tex, vec3 p, vec3 geomNormal) {
  vec3 blend = abs(geomNormal);
  blend = pow(blend, vec3(5.0));
  blend /= (blend.x + blend.y + blend.z);

  vec3 tX = texture(tex, p.yz * 0.5).rgb * 2.0 - 1.0;
  vec3 tY = texture(tex, p.xz * 0.5).rgb * 2.0 - 1.0;
  vec3 tZ = texture(tex, p.xy * 0.5).rgb * 2.0 - 1.0;

  tX.x *= sign(geomNormal.x);
  tY.x *= sign(geomNormal.y); 
  tZ.x *= sign(geomNormal.z); 

  tX = vec3(tX.xy + geomNormal.zy, abs(tX.z) * geomNormal.x);
  tY = vec3(tY.xy + geomNormal.xz, abs(tY.z) * geomNormal.y);
  tZ = vec3(tZ.xy + geomNormal.xy, abs(tZ.z) * geomNormal.z);

  vec3 nX = vec3(tX.z, tX.x, tX.y);
  vec3 nY = vec3(tY.x, tY.z, tY.y);
  vec3 nZ = vec3(tZ.x, tZ.y, tZ.z);

  return normalize(nX * blend.x + nY * blend.y + nZ * blend.z);
}

vec3 applyNormalMap(vec3 geomNormal, vec3 texNormal) {
  texNormal = texNormal * 2.0 - 1.0;
  vec3 up = abs(geomNormal.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
  vec3 tangent = normalize(cross(up, geomNormal));
  vec3 bitangent = cross(geomNormal, tangent);
  vec3 finalNormal = tangent * texNormal.x + bitangent * texNormal.y + geomNormal * texNormal.z;
  return normalize(finalNormal);
}

SDF map(vec3 p) {
  SDF sphere1;
  sphere1.distance = sdSphere(p - vec3(0.3, -0.23, 0.0), 0.1);
  sphere1.id = 10.0;

  SDF sphere2;
  sphere2.distance = sdSphere(p - vec3(0.0, -0.3, 0.0), 0.1);
  sphere2.id = 7.0;
  if (sphere2.distance < 0.05) {
    float sphere2_disp = textureLod(u_onyx_displacement, p.xz, 0.0).r;
    sphere2_disp *= 0.01;
    sphere2.distance = sphere2.distance - sphere2_disp;
  }

  SDF sphere3;
  sphere3.distance = sdSphere(p - vec3(-0.3, -0.3, 0.0), 0.1);
  sphere3.id = 2.0;

  SDF sun;
  // sun.distance = sdBox(p - vec3(0.0, 0.55, 0.5), vec3(0.2, 0.01, 0.2));
  sun.distance = sdBox(p - c_lightPos, vec3(c_lightSize.x, 0.01, c_lightSize.y));
  sun.id = 0.0;

  SDF ground;
  ground.distance = sdPlane(p, vec3(0.0, 1.0, 0.0), 0.4);
  ground.id = 9.0;
  if (ground.distance < 0.05) {
    float ground_disp = textureLod(u_ground_disp, p.xz, 0.0).r;
    ground_disp *= 0.005;
    ground.distance = ground.distance - ground_disp;
  }

  SDF back;
  back.distance = sdPlane(p, vec3(0.0, 0.0, 1.0), 0.5);
  back.id = 11.0;
  if (back.distance < 0.05) {
    float back_disp = textureLod(u_tile_displacement, p.xy, 0.0).r;
    back_disp *= 0.01;
    back.distance = back.distance - back_disp;
  }

  SDF left;
  left.distance = sdPlane(p, vec3(1.0, 0.0, 0.0), 0.5);
  left.id = 5.0;

  SDF right;
  right.distance = sdPlane(p, vec3(-1.0, 0.0, 0.0), 0.5);
  right.id = 6.0;

  SDF top;
  top.distance = sdPlane(p, vec3(0.0, -1.0, 0.0), 0.51);
  top.id = 4.0; // white

  SDF behind;
  behind.distance = sdPlane(p, vec3(0.0, 0.0, -1.0), 1.5);
  behind.id = 4.0;

  SDF result = opUnionID(sphere1, sphere2);
  result = opUnionID(result, sphere3);
  result = opUnionID(result, ground);
  result = opUnionID(result, back);
  result = opUnionID(result, left);
  result = opUnionID(result, right);
  result = opUnionID(result, top);
  result = opUnionID(result, behind);
  result = opUnionID(result, sun);

  return result;
}

vec3 getNormal(vec3 p) {
  vec2 e = vec2(EPSILON, 0.0);
  return normalize(vec3(map(p + e.xyy).distance - map(p - e.xyy).distance,
                        map(p + e.yxy).distance - map(p - e.yxy).distance,
                        map(p + e.yyx).distance - map(p - e.yyx).distance));
}

SDF rayMarch(vec3 ro, vec3 rd) {
  SDF hit, object;
  float t = 0.0;
  for (int i = 0; i < MAX_STEPS; i++) {
    vec3 p = ro + rd * t;
    hit = map(p); 
    float stepDist = max(abs(hit.distance) * 0.5, EPSILON * 0.5);
    t += stepDist;
    object.distance = hit.distance;
    object.id = hit.id;
    if (abs(hit.distance) < EPSILON || t > MAX_DIST)
      break;
  }
  object.distance = t;
  object.id = hit.id;
  return object;
}

Material getMaterial(SDF object, vec3 p, inout vec3 normal) {
  Material material = GetZeroedMaterial();
  switch (int(object.id)) {
    case 0:
      vec3 color = SRGBToLinear(u_tempColor);
      material.emissive = color * 20.0f;
      material.albedo = vec3(0);
      break;
    case 1:
      material.emissive = vec3(0.0, 0.0, 0.0);
      material.albedo = vec3(0.1, 0.7, 0.1);
      break;
    case 2:
      material.emissive = vec3(0.0, 0.0, 0.0);
      material.albedo = vec3(0.7, 0.7, 0.7);
      break;
    case 3:
      material.emissive = vec3(0.0, 0.0, 0.0);
      material.albedo = vec3(0.7, 0.1, 0.1);
      break;
    case 4:
      material.emissive = vec3(0.0, 0.0, 0.0);
      float stripe = mod(floor(p.x * 20.0), 2.0);
      material.albedo = mix(vec3(0.7f, 0.7f, 0.7f), vec3(0.2f, 0.2f, 0.2f), stripe);
      break;
    case 5:
      material.emissive = vec3(0.0, 0.0, 0.0);
      material.albedo = vec3(0.7f, 0.1f, 0.1f);
      break;
    case 6:
      material.emissive = vec3(0.0, 0.0, 0.0);
      material.albedo = vec3(0.1f, 0.7f, 0.1f);
      break;
    case 7:
      material.albedo = SRGBToLinear(triPlanar(u_onyx, p, normal).rgb);
      material.metallic = 0.0;
      material.roughness = triPlanar(u_onyx_roughness, p, normal).r;
      break;
    case 8:
      material.albedo = vec3(0.8, 0.1, 0.1); 
      material.metallic = 0.0; 
      material.roughness = 1.0; 
      break;
    case 9:
      vec3 texColor = texture(u_ground, p.xz).rgb;
      float texRough = texture(u_ground_roughness, p.xz).r;
      material.albedo = SRGBToLinear(texColor);
      material.roughness = texRough;
      material.metallic = 0.0;
      break;
    case 10:
      material.albedo = vec3(0.0, 0.0, 0.0);
      material.metallic = 0.0;
      material.transmission = 1.0;
      material.IOR = 1.3;
      material.roughness = 0.02;
      material.absorption = vec3(1.0, 0.0, 0.0);
      break;
    case 11:
      material.albedo = SRGBToLinear(texture(u_tile, p.xy).rgb);
      material.roughness = texture(u_tile_roughness, p.xy).r;
      material.metallic = 0.0;
      break;
    case 12:
      material.albedo = vec3(1.0, 1.0, 0.0);
      break;
    default:
      material.albedo = vec3(1.0, 0.0, 1.0); 
      break;
  }
  return material;
}

bool isVisibleToLight(vec3 origin, vec3 dir, float maxDist) {
  float t = EPSILON * 4.0;
  for (int i = 0; i < SHADOW_MAX_STEPS; ++i) {
    if (t >= maxDist) {
      return true;
    }
    vec3 p = origin + dir * t;
    SDF h = map(p);
    if (h.distance < EPSILON) {
      return h.id == 0.0;
    }
    t += max(abs(h.distance) * 0.95, EPSILON * 0.5);
  }
  return true;
}

vec3 sampleRectLight(vec3 hitPos, vec3 normal, Material mat, inout uint rngState) {
  // Glass/water shouldn't calculate direct diffuse shadows this way
  if (mat.transmission > 0.0) {
    return vec3(0.0);
  }

  // 1. Pick a random point on the bottom face of our rectangular light
  float rX = RandomValue(rngState) * 2.0 - 1.0;
  float rZ = RandomValue(rngState) * 2.0 - 1.0;
  
  // We offset Y by -0.01 to match the bottom face of the sdBox
  vec3 lightPoint = c_lightPos + vec3(rX * c_lightSize.x, -0.01, rZ * c_lightSize.y);
  
  // The normal of a ceiling light points straight down
  vec3 lightNormal = vec3(0.0, -1.0, 0.0); 

  vec3 toLight = lightPoint - hitPos;
  float dist2 = dot(toLight, toLight);
  if (dist2 <= 1e-5) {
    return vec3(0.0);
  }

  float dist = sqrt(dist2);
  vec3 wi = toLight / dist;

  // Cosine of angle between surface normal and light direction
  float ndotl = max(dot(normal, wi), 0.0);
  if (ndotl <= 0.0) {
    return vec3(0.0);
  }

  // 2. Cast a shadow ray
  if (!isVisibleToLight(hitPos + normal * c_rayPosNormalNudge, wi, dist - EPSILON * 4.0)) {
    return vec3(0.0);
  }

  // Cosine of angle between light normal and incoming ray
  float lightCos = max(dot(lightNormal, -wi), 0.0);
  if (lightCos <= 0.0) {
    return vec3(0.0); // The ray hit the side/back of the light box
  }

  // 3. Calculate the Probability Density Function (PDF)
  // Area of a rectangle = (width) * (height) -> (2 * half_x) * (2 * half_z)
  float area = (2.0 * c_lightSize.x) * (2.0 * c_lightSize.y);
  float pdfA = 1.0 / area;
  
  // Convert Area PDF to Solid Angle PDF
  float pdfW = pdfA * dist2 / max(lightCos, 1e-4);
  if (pdfW <= 0.0) {
    return vec3(0.0);
  }

  // 4. Calculate final contribution
  vec3 diffuseColor = mat.albedo * (1.0 - mat.metallic);
  vec3 f = diffuseColor * (1.0 / c_pi); // Lambertian BRDF
  vec3 Le = SRGBToLinear(u_tempColor) * c_lightEmissionStrength;

  return Le * f * ndotl / pdfW;
}

// vec3 sampleDirectSun(vec3 hitPos, vec3 normal, Material mat, inout uint rngState) {
//   if (mat.transmission > 0.0) {
//     return vec3(0.0);
//   }
//
//   vec3 lightDirRandom = RandomUnitVector(rngState);
//   vec3 lightPoint = c_sunPos + lightDirRandom * c_sunRadius;
//   vec3 toLight = lightPoint - hitPos;
//   float dist2 = dot(toLight, toLight);
//   if (dist2 <= 1e-5) {
//     return vec3(0.0);
//   }
//
//   float dist = sqrt(dist2);
//   vec3 wi = toLight / dist;
//   float ndotl = max(dot(normal, wi), 0.0);
//   if (ndotl <= 0.0) {
//     return vec3(0.0);
//   }
//
//   if (!isVisibleToLight(hitPos + normal * c_rayPosNormalNudge, wi, dist - EPSILON * 4.0)) {
//     return vec3(0.0);
//   }
//
//   vec3 lightNormal = normalize(lightPoint - c_sunPos);
//   float lightCos = max(dot(lightNormal, -wi), 0.0);
//   if (lightCos <= 0.0) {
//     return vec3(0.0);
//   }
//
//   float area = 4.0 * c_pi * c_sunRadius * c_sunRadius;
//   float pdfA = 1.0 / area;
//   float pdfW = pdfA * dist2 / max(lightCos, 1e-4);
//   if (pdfW <= 0.0) {
//     return vec3(0.0);
//   }
//
//   vec3 diffuseColor = mat.albedo * (1.0 - mat.metallic);
//   vec3 f = diffuseColor * (1.0 / c_pi);
//   vec3 Le = SRGBToLinear(u_tempColor) * c_sunEmissionStrength;
//
//   return Le * f * ndotl / pdfW;
// }

void getCurrentHit(vec3 rayOrigin, vec3 rayDirection,
                   inout SRayHitInfo hitInfo) {
  SDF object = rayMarch(rayOrigin, rayDirection);
  if (object.distance < MAX_DIST) {
    hitInfo.dist = object.distance;
    hitInfo.normal = getNormal(rayOrigin + rayDirection * object.distance);
    hitInfo.material = getMaterial(
        object, rayOrigin + rayDirection * object.distance, hitInfo.normal);

    if (dot(rayDirection, hitInfo.normal) > 0.0) {
      hitInfo.normal = -hitInfo.normal;
      hitInfo.fromInside = true;
    }
  } else {
    hitInfo.dist = c_superFar;
  }
}

vec3 getColorForRay(in vec3 startRayPos, in vec3 startRayDir,
                    inout uint rngState) {
  vec3 ret = vec3(0.0f, 0.0f, 0.0f);
  vec3 throughput = vec3(1.0f, 1.0f, 1.0f);
  vec3 rayPos = startRayPos;
  vec3 rayDir = startRayDir;
  bool prevWasDelta = true;

  for (int bounceIndex = 0; bounceIndex <= c_numBounces; ++bounceIndex) {
    SRayHitInfo hitInfo;
    hitInfo.material = GetZeroedMaterial();
    hitInfo.dist = c_superFar;
    hitInfo.fromInside = false;
    getCurrentHit(rayPos, rayDir, hitInfo);

    if (hitInfo.dist == c_superFar) {
      break;
    }

    if (prevWasDelta || bounceIndex == 0) {
      ret += throughput * hitInfo.material.emissive;
    }

    if (length(hitInfo.material.emissive) < 1e-6 &&
        hitInfo.material.transmission <= 0.0) {
      vec3 hitPosDirect = rayPos + rayDir * hitInfo.dist;
      ret += throughput * sampleRectLight(hitPosDirect, hitInfo.normal, hitInfo.material,
                                          rngState);
    }

    if (hitInfo.fromInside)
      throughput *= exp(-hitInfo.material.absorption * hitInfo.dist);

    float n1 = hitInfo.fromInside ? hitInfo.material.IOR : 1.0f;
    float n2 = hitInfo.fromInside ? 1.0f : hitInfo.material.IOR;
    float eta = n1 / n2;

    float fresnelChance = FresnelReflectAmount(n1, n2, hitInfo.normal, rayDir, 0.04f, 1.0f);
    float P_reflect = mix(fresnelChance, 1.0f, hitInfo.material.metallic);
    float P_transmit = hitInfo.material.transmission * (1.0f - P_reflect);
    float P_diffuse = 1.0f - (P_reflect + P_transmit);

    P_reflect = max(P_reflect, 0.001f);
    P_transmit = max(P_transmit, 0.001f);
    P_diffuse = max(P_diffuse, 0.001f);

    float roll = RandomValue(rngState);
    vec3 hitPos = rayPos + rayDir * hitInfo.dist;

    if (roll < P_reflect) {
      vec3 H = ImportanceSampleGGX(
          vec2(RandomValue(rngState), RandomValue(rngState)),
          hitInfo.normal, hitInfo.material.roughness);
      rayDir = reflect(rayDir, H);
      rayPos = hitPos + hitInfo.normal * c_rayPosNormalNudge;

      vec3 reflectionColor = mix(vec3(1.0f), hitInfo.material.albedo, hitInfo.material.metallic);
      throughput *= reflectionColor;
      throughput /= P_reflect;
      prevWasDelta = hitInfo.material.roughness <= 0.02;
    } else if (roll < P_reflect + P_transmit) {
      vec3 H = ImportanceSampleGGX(
          vec2(RandomValue(rngState), RandomValue(rngState)),
          hitInfo.normal, hitInfo.material.roughness);
      rayDir = refract(rayDir, H, eta);

      rayPos = hitPos - hitInfo.normal * c_rayPosNormalNudge;

      throughput /= P_transmit;
      prevWasDelta = hitInfo.material.roughness <= 0.02;
    } else {
      rayDir = normalize(hitInfo.normal + RandomUnitVector(rngState));
      rayPos = hitPos + hitInfo.normal * c_rayPosNormalNudge;

      vec3 diffuseColor = hitInfo.material.albedo * (1.0f - hitInfo.material.metallic);
      throughput *= diffuseColor;
      throughput /= P_diffuse;
      prevWasDelta = false;
    }

    float p = max(throughput.r, max(throughput.g, throughput.b));
    if (RandomValue(rngState) > p) {
      break;
    }
    throughput *= 1.0f / p;
  }

  return ret;
}

mat3 getCam(vec3 ro, vec3 lookAt) {
  vec3 camF = normalize(vec3(lookAt - ro));
  vec3 camR = normalize(cross(vec3(0, 1, 0), camF));
  vec3 camU = cross(camF, camR);
  return mat3(camR, camU, camF);
}

void applyRotation(inout vec3 ro) {
  pR(ro.yz, u_cameraRot.y);
  pR(ro.xz, u_cameraRot.x);
}

void main() {
  uint state =
    uint(uint(gl_FragCoord.x) * uint(1973) +
        uint(gl_FragCoord.y) * uint(9277) + uint(u_frame) * uint(26699)) |
      uint(1);

  state = NextRandom(state);
  state = NextRandom(state);
  vec3 accumulated_color = vec3(0.0);
  int samples_per_frame = u_spf;
  float focusDist = u_focusDist; 
  float aperture = u_aperture; 

  for (int i = 0; i < samples_per_frame; i++) {
    vec2 jitter = vec2(RandomValue(state), RandomValue(state)) - 0.5;
    vec2 uv = (gl_FragCoord.xy + jitter) / u_resolution.xy;
    vec2 pixelTarget2D = (uv * 2.0f) - 1.0f;
    float aspectRatio = u_resolution.x / u_resolution.y;

    float cameraDistance = 1.0f / tan(c_FOVDegrees * 0.5f * c_pi / 180.0f);

    vec3 rayDir = normalize(
        vec3(pixelTarget2D.x * aspectRatio, pixelTarget2D.y, -cameraDistance));

    pR(rayDir.yz, u_cameraRot.y); 
    pR(rayDir.xz, u_cameraRot.x); 
    
    vec3 focal_point = u_cameraPos + rayDir * focusDist;

    vec3 camRight = vec3(1.0, 0.0, 0.0);
    pR(camRight.yz, u_cameraRot.y);
    pR(camRight.xz, u_cameraRot.x);

    vec3 camUp = vec3(0.0, 1.0, 0.0);
    pR(camUp.yz, u_cameraRot.y);
    pR(camUp.xz, u_cameraRot.x);

    vec2 lensJitter = RandomPointInCircle(state) * aperture;
    
    Ray ray;
    ray.origin = u_cameraPos + camRight * lensJitter.x + camUp * lensJitter.y;
    ray.direction = normalize(focal_point - ray.origin);

    accumulated_color += getColorForRay(ray.origin, ray.direction, state);
  }

  vec3 current_average = accumulated_color / float(samples_per_frame);

  vec4 lastFrameColor = texture(u_pass1, gl_FragCoord.xy / u_resolution.xy);
  float blend = 1.0 / float(u_frame + 1);

  vec3 color = mix(lastFrameColor.rgb, current_average, blend);

  fragColor = vec4(color, 1.0);
}
