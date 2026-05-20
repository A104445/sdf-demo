#version 410

uniform vec2 u_resolution; // viewport size in pixels (width, height)
uniform int u_frame; // frame increment counter
uniform vec2 u_mouse;
uniform vec2 u_cameraRot;
uniform vec3 u_cameraPos;
uniform float u_time;
uniform vec3 u_tempColor;

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

// ACES tone mapping curve fit to go from HDR to LDR
// https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
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
  // Schlick aproximation
  float r0 = (n1 - n2) / (n1 + n2);
  r0 *= r0;
  float cosX = -dot(normal, incident);
  if (n1 > n2) {
    float n = n1 / n2;
    float sinT2 = n * n * (1.0 - cosX * cosX);
    // Total internal reflection
    if (sinT2 > 1.0)
      return f90;
    cosX = sqrt(1.0 - sinT2);
  }
  float x = 1.0 - cosX;
  float ret = r0 + (1.0 - r0) * x * x * x * x * x;

  // adjust reflect multiplier for object reflectivity
  return mix(f0, f90, ret);
}

const int MAX_STEPS = 256;
const float EPSILON = 0.001;
const float MAX_DIST = 500.0;
const float c_FOVDegrees = 90.0f;
const float c_pi = 3.14159265359f;
const float c_twopi = 2.0f * c_pi;
// the farthest we look for path tracing ray hits
const float c_superFar = 10000.0f;
const int c_numBounces = 8;

struct Ray {
  vec3 origin;
  vec3 direction;
  vec3 target;
};

struct Material {
  vec3 albedo; // Base color (surface reflections/F0)
  float roughness; // 0.0 = smooth, 1.0 = frosted
  float metallic;
  vec3 emissive;

  // transparent materials
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

float opUnion(float a, float b) {
  return min(a, b);
}

SDF opUnionID(SDF res1, SDF res2) {
  if (res1.distance < res2.distance) {
    return res1;
  } else {
    return res2;
  }
}

// PCG (permuted congruential generator). Thanks to:
// www.pcg-random.org and www.shadertoy.com/view/XlGcRh
uint NextRandom(inout uint state) {
  state = state * 747796405u + 2891336453u;
  uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  uint result = (word >> 22u) ^ word;
  return result;
}

float RandomValue(inout uint state) {
  return float(NextRandom(state)) / 4294967295.0; // 2^32 - 1
}

// Random value in normal distribution (with mean=0 and sd=1)
float RandomValueNormalDistribution(inout uint state) {
  float u1 = RandomValue(state);
  float u2 = RandomValue(state);

  // Safeguard against log(0) which is undefined/infinity
  // We use a tiny offset (epsilon)
  u1 = max(u1, 1e-7);

  float theta = 2.0 * 3.14159265359 * u2;
  float rho = sqrt(-2.0 * log(u1));
  return rho * cos(theta);
}

// Calculate a random direction
vec3 RandomDirection(inout uint state) {
  // Thanks to https://math.stackexchange.com/a/1585996
  float x = RandomValueNormalDistribution(state);
  float y = RandomValueNormalDistribution(state);
  float z = RandomValueNormalDistribution(state);
  return normalize(vec3(x, y, z));
}

// Cook torrance things

float GeometrySchlickGGX(float NdotV, float roughness) {
  float a = roughness;
  float k = (a * a) / 2.0;

  float nom = NdotV;
  float denom = NdotV * (1.0 - k) + k;

  return nom / denom;
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
  float NdotV = max(dot(N, V), 0.0);
  float NdotL = max(dot(N, L), 0.0);
  float ggx2 = GeometrySchlickGGX(NdotV, roughness);
  float ggx1 = GeometrySchlickGGX(NdotL, roughness);

  return ggx1 * ggx2;
}

vec3 ImportanceSampleGGX(vec2 Xi, vec3 N, float roughness) {
  // Disney/Epic Games roughness remapping
  float a = roughness * roughness;

  // Convert our random numbers into spherical coordinates (phi and theta)
  // based on the exact GGX distribution curve
  float phi = 2.0 * c_pi * Xi.x;
  float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a * a - 1.0) * Xi.y));
  float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

  // Convert spherical coordinates to tangent-space vector
  vec3 H;
  H.x = cos(phi) * sinTheta;
  H.y = sin(phi) * sinTheta;
  H.z = cosTheta;

  // We need to rotate this tangent-space vector to align with our actual
  // surface Normal We create a Coordinate System (Tangent, Bitangent, Normal)
  vec3 up = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
  vec3 tangent = normalize(cross(up, N));
  vec3 bitangent = cross(N, tangent);

  // Transform H from tangent space to world space
  vec3 sampleVec = tangent * H.x + bitangent * H.y + N * H.z;
  return normalize(sampleVec);
}

float dot2(vec3 v) {
  return dot(v, v);
}

float sdSphere(vec3 p, float r) {
  return length(p) - r;
}

uint wang_hash(inout uint seed) {
  seed = uint(seed ^ uint(61)) ^ uint(seed >> uint(16));
  seed *= uint(9);
  seed = seed ^ (seed >> 4);
  seed *= uint(0x27d4eb2d);
  seed = seed ^ (seed >> 15);
  return seed;
}

float RandomFloat01(inout uint state) {
  return float(wang_hash(state)) / 4294967296.0;
}

vec3 RandomUnitVector(inout uint state) {
  float z = RandomFloat01(state) * 2.0f - 1.0f;
  float a = RandomFloat01(state) * c_twopi;
  float r = sqrt(1.0f - z * z);
  float x = r * cos(a);
  float y = r * sin(a);
  return vec3(x, y, z);
}

float sdPlane(vec3 p, vec3 n, float h) {
  // n must be normalized
  return dot(p, n) + h;
}

float sdBox(vec3 p, vec3 b) {
  vec3 q = abs(p) - b;
  return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
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
    texture(tex, p.yz * 0.5 + 0.5) * normal.x)
  .rgb;
}

vec3 triPlanarNormal(sampler2D tex, vec3 p, vec3 geomNormal) {
  // 1. Calculate the blend weights based on the macro normal
  vec3 blend = abs(geomNormal);
  blend = pow(blend, vec3(5.0));
  blend /= (blend.x + blend.y + blend.z);

  // 2. Sample the normal map 3 times and unpack to [-1, 1] vectors
  vec3 tX = texture(tex, p.yz * 0.5).rgb * 2.0 - 1.0;
  vec3 tY = texture(tex, p.xz * 0.5).rgb * 2.0 - 1.0;
  vec3 tZ = texture(tex, p.xy * 0.5).rgb * 2.0 - 1.0;

  // 3. Prevent inverted bumps on the "back" sides of the SDF.
  // We flip the tangent (X) of the normal map based on the facing direction.
  tX.x *= sign(geomNormal.x);
  tY.x *= sign(geomNormal.y); // Flipped X instead of Y
  tZ.x *= sign(geomNormal.z); // Flipped X instead of Z

  tX = vec3(tX.xy + geomNormal.zy, abs(tX.z) * geomNormal.x);
  tY = vec3(tY.xy + geomNormal.xz, abs(tY.z) * geomNormal.y);
  tZ = vec3(tZ.xy + geomNormal.xy, abs(tZ.z) * geomNormal.z);

  // 5. Swizzle the vectors to align with world space based on UV mapping!
  // p.yz (U=Y, V=Z) -> TexX=WorldY, TexY=WorldZ
  vec3 nX = vec3(tX.z, tX.x, tX.y);
  // p.xz (U=X, V=Z) -> TexX=WorldX, TexY=WorldZ
  vec3 nY = vec3(tY.x, tY.z, tY.y);
  // p.xy (U=X, V=Y) -> TexX=WorldX, TexY=WorldY
  vec3 nZ = vec3(tZ.x, tZ.y, tZ.z);

  // 6. Blend and return the final bent normal
  return normalize(nX * blend.x + nY * blend.y + nZ * blend.z);
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
  sun.distance = sdBox(p - vec3(0.0, 0.5, 0.5), vec3(0.2, 0.01, 0.2));
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
    hit = map(p); // Distance to nearest object
    // really need to understand why mult this by 0.98 fixes the pattern
    t += abs(hit.distance) * 0.5;
    object.distance = hit.distance;
    object.id = hit.id;
    if (abs(hit.distance) < EPSILON || t > MAX_DIST)
      break;
  }
  object.distance = t;
  object.id = hit.id;
  return object;
}

vec3 applyNormalMap(vec3 geomNormal, vec3 texNormal) {
  // Convert texture RGB from [0.0, 1.0] to direction vectors [-1.0, 1.0]
  texNormal = texNormal * 2.0 - 1.0;

  // Build the TBN (Tangent, Bitangent, Normal) coordinate system
  vec3 up =
    abs(geomNormal.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
  vec3 tangent = normalize(cross(up, geomNormal));
  vec3 bitangent = cross(geomNormal, tangent);

  // Rotate the texture's bump direction to match the geometry's surface
  vec3 finalNormal = tangent * texNormal.x + bitangent * texNormal.y +
      geomNormal * texNormal.z;

  return normalize(finalNormal);
}

Material getMaterial(SDF object, vec3 p, inout vec3 normal) {
  Material material = GetZeroedMaterial();
  switch (int(object.id)) {
    // white light
    case 0:
    // material.emissive = vec3(1, 0.7529422167760779, 0.5775804404296506)
    // * 20.0f; material.emissive = vec3(1.0) * 20.0f;
    vec3 color = SRGBToLinear(u_tempColor);
    material.emissive = color * 20.0f;
    material.albedo = vec3(0);
    break;
    //
    case 1:
    material.emissive = vec3(0.0, 0.0, 0.0);
    material.albedo = vec3(0.1, 0.7, 0.1);
    break;
    // white
    case 2:
    material.emissive = vec3(0.0, 0.0, 0.0);
    material.albedo = vec3(0.7, 0.7, 0.7);
    break;
    // red
    case 3:
    material.emissive = vec3(0.0, 0.0, 0.0);
    material.albedo = vec3(0.7, 0.1, 0.1);
    break;
    // back wall
    case 4:
    material.emissive = vec3(0.0, 0.0, 0.0);
    float stripe = mod(floor(p.x * 20.0), 2.0);
    material.albedo =
      mix(vec3(0.7f, 0.7f, 0.7f), vec3(0.2f, 0.2f, 0.2f), stripe);
    // material.albedo = vec3(0.7, 0.7, 0.7);
    break;
    // left wall
    case 5:
    material.emissive = vec3(0.0, 0.0, 0.0);
    material.albedo = vec3(0.7f, 0.1f, 0.1f);
    break;
    // right wall
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
    material.albedo = vec3(0.8, 0.1, 0.1); // Red
    material.metallic = 0.0; // Not a metal
    material.roughness = 1.0; // Quite rough
    break;
    case 9:
    vec3 texColor = texture(u_ground, p.xz).rgb;
    float texRough = texture(u_ground_roughness, p.xz).r;
    material.albedo = SRGBToLinear(texColor);
    material.roughness = texRough;
    material.metallic = 0.0;

    // normal = triPlanarNormal(u_ground_normal, p, normal);

    break;
    // glass
    case 10:
    material.albedo = vec3(0.0, 0.0, 0.0);
    // material.albedo = vec3(1.0, 0.0, 0.0);
    material.metallic = 0.0;
    material.transmission = 1.0;
    material.IOR = 1.3;
    material.roughness = 0.02;
    // material.absorption = vec3(0.0);
    material.absorption = vec3(1.0, 1.0, 1.0);
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
    material.albedo = vec3(1.0, 0.0, 1.0); // bright magenta for unhandled ids
    break;
  }
  return material;
}

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
  // initialize
  vec3 ret = vec3(0.0f, 0.0f, 0.0f);
  vec3 throughput = vec3(1.0f, 1.0f, 1.0f);
  vec3 rayPos = startRayPos;
  vec3 rayDir = startRayDir;

  for (int bounceIndex = 0; bounceIndex <= c_numBounces; ++bounceIndex) {
    // shoot a ray out into the world
    SRayHitInfo hitInfo;
    hitInfo.material = GetZeroedMaterial();
    hitInfo.dist = c_superFar;
    hitInfo.fromInside = false;
    getCurrentHit(rayPos, rayDir, hitInfo);

    // if the ray missed, we are done
    if (hitInfo.dist == c_superFar) {
      break;
    }

    ret += throughput * hitInfo.material.emissive;

    // Beers law for volume absorption
    if (hitInfo.fromInside)
      throughput *= exp(-hitInfo.material.absorption * hitInfo.dist);

    float n1 = hitInfo.fromInside ? hitInfo.material.IOR : 1.0f;
    float n2 = hitInfo.fromInside ? 1.0f : hitInfo.material.IOR;
    float eta = n1 / n2;

    // Base reflection chance (0.04 is the standard F0 for most non-metals)
    float fresnelChance =
      FresnelReflectAmount(n1, n2, hitInfo.normal, rayDir, 0.04f, 1.0f);
    // Metals force 100% reflection. Dielectrics reflect based on Fresnel.
    float P_reflect = mix(fresnelChance, 1.0f, hitInfo.material.metallic);
    // Transmission (glass) only happens if the ray didn't reflect
    float P_transmit = hitInfo.material.transmission * (1.0f - P_reflect);
    // Diffuse happens if it didn't reflect and didn't transmit
    float P_diffuse = 1.0f - (P_reflect + P_transmit);

    // Numerical safeguard
    P_reflect = max(P_reflect, 0.001f);
    P_transmit = max(P_transmit, 0.001f);
    P_diffuse = max(P_diffuse, 0.001f);

    float roll = RandomFloat01(rngState);
    vec3 hitPos = rayPos + rayDir * hitInfo.dist;

    if (roll < P_reflect) {
      // PATH A: SPECULAR REFLECTION
      vec3 V = -rayDir; // View vector is the reverse of the incoming ray

      vec3 H = ImportanceSampleGGX(
          vec2(RandomFloat01(rngState), RandomFloat01(rngState)),
          hitInfo.normal, hitInfo.material.roughness);

      // The new outgoing ray direction (L)
      vec3 L = reflect(rayDir, H);
      rayDir = L;
      rayPos = hitPos + hitInfo.normal * c_rayPosNormalNudge;

      // Calculate the Geometry/Shadowing term
      float G = GeometrySmith(hitInfo.normal, V, L, hitInfo.material.roughness);

      // The Monte Carlo estimator weight when importance sampling GGX
      float NdotV = max(dot(hitInfo.normal, V), 0.001);
      float NdotH = max(dot(hitInfo.normal, H), 0.001);
      float VdotH = max(dot(V, H), 0.001);

      // The D term and 4*NdotL cancel out with the PDF
      float weight = (G * VdotH) / (NdotV * NdotH);

      // Dielectric reflections are white, Metal reflections are tinted by albedo
      vec3 reflectionColor =
        mix(vec3(1.0f), hitInfo.material.albedo, hitInfo.material.metallic);

      throughput *= reflectionColor * weight;
      throughput /= P_reflect;
    } else if (roll < P_reflect + P_transmit) {

      // PATH B: TRANSMISSION (GLASS/WATER)
      vec3 H = ImportanceSampleGGX(
          vec2(RandomFloat01(rngState), RandomFloat01(rngState)),
          hitInfo.normal, hitInfo.material.roughness);

      rayDir = refract(rayDir, H, eta);

      // Nudge the ray *into* the object by subtracting the normal
      rayPos = hitPos - hitInfo.normal * c_rayPosNormalNudge;

      throughput /= P_transmit;
    } else {
      // PATH C: DIFFUSE
      rayDir = normalize(hitInfo.normal + RandomUnitVector(rngState));
      rayPos = hitPos + hitInfo.normal * c_rayPosNormalNudge;

      // Metals have no diffuse color.
      vec3 diffuseColor =
        hitInfo.material.albedo * (1.0f - hitInfo.material.metallic);
      throughput *= diffuseColor;
      throughput /= P_diffuse;
    }

    // 4. Russian Roulette (Path Termination)
    float p = max(throughput.r, max(throughput.g, throughput.b));
    if (RandomFloat01(rngState) > p) {
      break;
    }
    throughput *= 1.0f / p;
  }

  return ret;
}

void pR(inout vec2 p, float a) {
  p = cos(a) * p + sin(a) * vec2(p.y, -p.x);
}

mat3 getCam(vec3 ro, vec3 lookAt) {
  vec3 camF = normalize(vec3(lookAt - ro));
  vec3 camR = normalize(cross(vec3(0, 1, 0), camF));
  vec3 camU = cross(camF, camR);
  return mat3(camR, camU, camF);
}

void applyRotation(inout vec3 ro) {
  // Pitch (up/down) around the XZ plane
  pR(ro.yz, u_cameraRot.y);
  // Yaw (left/right) around the vertical axis
  pR(ro.xz, u_cameraRot.x);
}

void main() {
  // RNG setup
  uint state =
    uint(uint(gl_FragCoord.x) * uint(1973) +
        uint(gl_FragCoord.y) * uint(9277) + uint(u_frame) * uint(26699)) |
      uint(1);
  vec3 accumulated_color = vec3(0.0);
  int samples_per_frame = u_spf;

  for (int i = 0; i < samples_per_frame; i++) {
    vec2 jitter = vec2(RandomFloat01(state), RandomFloat01(state)) - 0.5;
    vec2 uv = (gl_FragCoord.xy + jitter) / u_resolution.xy;
    vec2 pixelTarget2D = (uv * 2.0f) - 1.0f;
    float aspectRatio = u_resolution.x / u_resolution.y;

    // calculate the camera distance
    float cameraDistance = 1.0f / tan(c_FOVDegrees * 0.5f * c_pi / 180.0f);

    Ray ray;
    // 1. The origin is strictly controlled by WASD
    ray.origin = u_cameraPos;

    // 2. Base ray direction (pointing down the -Z axis like a standard camera)
    vec3 rayDir = normalize(
        vec3(pixelTarget2D.x * aspectRatio, pixelTarget2D.y, -cameraDistance));

    // 3. Rotate the ray itself based on mouse movement
    pR(rayDir.yz, u_cameraRot.y); // Pitch up/down
    pR(rayDir.xz, u_cameraRot.x); // Yaw left/right

    ray.direction = rayDir;

    accumulated_color += getColorForRay(ray.origin, ray.direction, state);
  }

  vec3 current_average = accumulated_color / float(samples_per_frame);

  // average the frames together
  vec4 lastFrameColor = texture(u_pass1, gl_FragCoord.xy / u_resolution.xy);

  // NEW MATH:
  // We need to tell the blend that we just added 64 samples, not 1.
  // If we use u_frame, it's much safer:
  float blend = 1.0 / float(u_frame + 1);

  vec3 color = mix(lastFrameColor.rgb, current_average, blend);

  // show the result - use a solid 1.0 for Alpha so Main pass can see it!
  fragColor = vec4(color, 1.0);
}
