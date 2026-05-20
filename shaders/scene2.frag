#version 410

uniform vec2 u_resolution; // viewport size in pixels (width, height)
uniform int u_frame;       // frame increment counter
uniform vec2 u_mouse;
uniform vec2 u_cameraRot;
uniform vec3 u_cameraPos;
uniform float u_time;
uniform vec3 u_tempColor;
uniform float u_focusDist;
uniform float u_aperture;


uniform sampler2D u_pass1;

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

const int MAX_STEPS = 192;
const float EPSILON = 0.001;
const float MAX_DIST = 500.0;
const float c_FOVDegrees = 90.0f;
const float c_pi = 3.14159265359f;
const float c_twopi = 2.0f * c_pi;
// the farthest we look for path tracing ray hits
const float c_superFar = 10000.0f;
const int c_numBounces = 8;
const int SHADOW_MAX_STEPS = 96;
const vec3 c_sunPos = vec3(100.0, 100.0, 0.0);
const float c_sunRadius = 10.0;
const float c_sunEmissionStrength = 500.0;

struct Ray {
  vec3 origin;
  vec3 direction;
  vec3 target;
};

struct Material {
  vec3 albedo;     // Base color (surface reflections/F0)
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

float opUnion(float a, float b) { return min(a, b); }

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

// uint wang_hash(inout uint seed) {
//   seed = uint(seed ^ uint(61)) ^ uint(seed >> uint(16));
//   seed *= uint(9);
//   seed = seed ^ (seed >> 4);
//   seed *= uint(0x27d4eb2d);
//   seed = seed ^ (seed >> 15);
//   return seed;
// }
//
// float RandomValue(inout uint state) {
//   return float(wang_hash(state)) / 4294967296.0;
// }

vec3 RandomUnitVector(inout uint state) {
  float z = RandomValue(state) * 2.0f - 1.0f;
  float a = RandomValue(state) * c_twopi;
  float r = sqrt(1.0f - z * z);
  float x = r * cos(a);
  float y = r * sin(a);
  return vec3(x, y, z);
}
// SDF definitions
float dot2(vec3 v) {
    return dot(v, v);
}

float vmax(vec2 v) {
    return max(v.x, v.y);
}

float vmax(vec3 v) {
    return max(max(v.x, v.y), v.z);
}

float vmax(vec4 v) {
    return max(max(v.x, v.y), max(v.z, v.w));
}

void pR(inout vec2 p, float a) {
    p = cos(a) * p + sin(a) * vec2(p.y, -p.x);
}

// Sign function that doesn't return 0
float sgn(float x) {
    if (x < 0.0) return -1.0;
    if (x > 0.0) return 1.0;
    return 0.0;
}

vec2 sgn(vec2 v) {
    return vec2(
        (v.x < 0.0) ? -1.0 : 1.0,
        (v.y < 0.0) ? -1.0 : 1.0
    );
}

float fBox2(vec2 p, vec2 b) {
    vec2 d = abs(p) - b;
    return length(max(d, vec2(0))) + vmax(min(d, vec2(0)));
}

float sdSphere(vec3 p, float r)
{
    return length(p) - r;
}

float sdBox(vec3 p, vec3 b)
{
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

float sdPlane(vec3 p, vec3 n, float h)
{
    // n must be normalized
    return dot(p, n) + h;
}

// Repeat space along one axis. Use like this to repeat along the x axis:
// <float cell = pMod1(p.x,5);> - using the return value is optional.
float pMod1(inout float p, float size) {
    float halfsize = size * 0.5;
    float c = floor((p + halfsize) / size);
    p = mod(p + halfsize, size) - halfsize;
    return c;
}

float sdCappedCylinder(vec3 p, float r, float h)
{
    vec2 d = abs(vec2(length(p.xz), p.y)) - vec2(r, h);
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}

// Mirror at an axis-aligned plane which is at a specified distance <dist> from the origin.
float pMirror(inout float p, float dist) {
    float s = sgn(p);
    p = abs(p) - dist;
    return s;
}

// Mirror in both dimensions and at the diagonal, yielding one eighth of the space.
// translate by dist before mirroring.
vec2 pMirrorOctant(inout vec2 p, vec2 dist) {
    vec2 s = sgn(p);
    pMirror(p.x, dist.x);
    pMirror(p.y, dist.y);
    if (p.y > p.x)
        p.xy = p.yx;
    return s;
}

SDF opDifferenceID(SDF res1, SDF res2) {
    return (res1.distance > -res2.distance) ? res1 : SDF(-res2.distance, res2.id);
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

SDF map(vec3 p) {
    SDF sun;
    sun.distance = sdSphere(p - c_sunPos, c_sunRadius);
    sun.id = 0.0;

    //Center Sphere
    SDF sphere;
    sphere.distance = sdSphere(p, 5.0);
    sphere.id = 1.0;

    //Walls
    pMirrorOctant(p.xz, vec2(50, 50));
    p.x = -abs(p.x) + 20;
    pMod1(p.z, 15);
    SDF wallBox;
    wallBox.distance = sdBox(p, vec3(3.0, 9.0, 4.0));
    wallBox.id = 1.0;
    SDF wallcylinder;
    vec3 pc = p;
    pc.y -= 9.0;
    wallcylinder.distance = sdCappedCylinder(pc.yxz, 4.0, 3.0);
    wallcylinder.id = 1.0;
    SDF infBox;
    infBox.distance = fBox2(p.xy, vec2(1.0, 15.0));
    infBox.id = 1.0;
    SDF walls = opUnionID(wallBox, wallcylinder);
    walls = opDifferenceID(infBox, walls);

    //Roof
    vec3 pr = p;
    pr.y -= 15.0;
    pR(pr.xy, 0.6);
    pr.x -= 18.0;
    SDF roof;
    roof.distance = fBox2(pr.xy, vec2(20.0, 0.3));
    roof.id = 1.0;

    SDF ground;
    ground.distance = sdPlane(p, vec3(0.0, 1.0, 0.0), 14);
    ground.id = 1.0;

    SDF result = opUnionID(walls, ground);
    result = opUnionID(result, roof);
    result = opUnionID(result, sphere);

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
    // Keep a small minimum step to avoid stalls near surfaces while marching faster in open space.
    float stepDist = max(abs(hit.distance) * 0.9, EPSILON * 0.5);
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

Material getMaterial(SDF object, vec3 p) {
  Material material = GetZeroedMaterial();
  switch (int(object.id)) {
  // white light
  case 0:
    // material.emissive = vec3(1, 0.7529422167760779, 0.5775804404296506)
    // * 20.0f; material.emissive = vec3(1.0) * 20.0f;
    vec3 color = SRGBToLinear(u_tempColor);
    material.emissive = color * 500.0f;
    material.albedo = vec3(0);
    break;
  //
  case 2:
    material.emissive = vec3(0.0, 0.0, 0.0);
    material.albedo = vec3(0.1, 0.7, 0.1);
    break;
  // white
  case 1:
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
    material.albedo = vec3(1.0, 0.84, 0.0); // Gold color
    material.metallic = 1.0;                // Fully metal
    material.roughness = 0.1;               // Very smooth
    break;
  case 8:
    material.albedo = vec3(0.8, 0.1, 0.1); // Red
    material.metallic = 0.0;               // Not a metal
    material.roughness = 1.0;              // Quite rough
    break;
  default:
    material.albedo = vec3(1.0, 0.0, 1.0); // bright magenta for unhandled ids
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

vec3 sampleDirectSun(vec3 hitPos, vec3 normal, Material mat, inout uint rngState) {
  if (mat.transmission > 0.0) {
    return vec3(0.0);
  }

  vec3 lightDirRandom = RandomUnitVector(rngState);
  vec3 lightPoint = c_sunPos + lightDirRandom * c_sunRadius;
  vec3 toLight = lightPoint - hitPos;
  float dist2 = dot(toLight, toLight);
  if (dist2 <= 1e-5) {
    return vec3(0.0);
  }

  float dist = sqrt(dist2);
  vec3 wi = toLight / dist;
  float ndotl = max(dot(normal, wi), 0.0);
  if (ndotl <= 0.0) {
    return vec3(0.0);
  }

  if (!isVisibleToLight(hitPos + normal * c_rayPosNormalNudge, wi, dist - EPSILON * 4.0)) {
    return vec3(0.0);
  }

  vec3 lightNormal = normalize(lightPoint - c_sunPos);
  float lightCos = max(dot(lightNormal, -wi), 0.0);
  if (lightCos <= 0.0) {
    return vec3(0.0);
  }

  float area = 4.0 * c_pi * c_sunRadius * c_sunRadius;
  float pdfA = 1.0 / area;
  float pdfW = pdfA * dist2 / max(lightCos, 1e-4);
  if (pdfW <= 0.0) {
    return vec3(0.0);
  }

  vec3 diffuseColor = mat.albedo * (1.0 - mat.metallic);
  vec3 f = diffuseColor * (1.0 / c_pi);
  vec3 Le = SRGBToLinear(u_tempColor) * c_sunEmissionStrength;

  return Le * f * ndotl / pdfW;
}

void getCurrentHit(vec3 rayOrigin, vec3 rayDirection,
                   inout SRayHitInfo hitInfo) {
  SDF object = rayMarch(rayOrigin, rayDirection);
  if (object.distance < MAX_DIST) {
    hitInfo.dist = object.distance;
    hitInfo.normal = getNormal(rayOrigin + rayDirection * object.distance);
    hitInfo.material =
        getMaterial(object, rayOrigin + rayDirection * object.distance);

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
  bool prevWasDelta = true;

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

    if (prevWasDelta || bounceIndex == 0) {
      ret += throughput * hitInfo.material.emissive;
    }

    if (length(hitInfo.material.emissive) < 1e-6 &&
      hitInfo.material.transmission <= 0.0) {
      vec3 hitPosDirect = rayPos + rayDir * hitInfo.dist;
      ret += throughput * sampleDirectSun(hitPosDirect, hitInfo.normal, hitInfo.material,
          rngState);
    }

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

    float roll = RandomValue(rngState);
    vec3 hitPos = rayPos + rayDir * hitInfo.dist;

    if (roll < P_reflect) {
      // PATH A: SPECULAR REFLECTION
      vec3 H = ImportanceSampleGGX(
          vec2(RandomValue(rngState), RandomValue(rngState)),
          hitInfo.normal, hitInfo.material.roughness);
      rayDir = reflect(rayDir, H);
      rayPos = hitPos + hitInfo.normal * c_rayPosNormalNudge;

      // Dielectric reflections are white, Metal reflections are tinted by the
      // albedo
      vec3 reflectionColor =
          mix(vec3(1.0f), hitInfo.material.albedo, hitInfo.material.metallic);
      throughput *= reflectionColor;
      throughput /= P_reflect;
      prevWasDelta = hitInfo.material.roughness <= 0.02;
    } else if (roll < P_reflect + P_transmit) {
      // PATH B: TRANSMISSION (GLASS/WATER)
      vec3 H = ImportanceSampleGGX(
          vec2(RandomValue(rngState), RandomValue(rngState)),
          hitInfo.normal, hitInfo.material.roughness);
      rayDir = refract(rayDir, H, eta);

      // Nudge the ray *into* the object
      rayPos = hitPos - hitInfo.normal * c_rayPosNormalNudge;

      // Note: We don't apply surface color here. Color comes from volume
      // absorption.
      throughput /= P_transmit;
      prevWasDelta = hitInfo.material.roughness <= 0.02;
    } else {
      // PATH C: DIFFUSE
      rayDir = normalize(hitInfo.normal + RandomUnitVector(rngState));
      rayPos = hitPos + hitInfo.normal * c_rayPosNormalNudge;

      // Metals have no diffuse color.
      vec3 diffuseColor =
          hitInfo.material.albedo * (1.0f - hitInfo.material.metallic);
      throughput *= diffuseColor;
      throughput /= P_diffuse;
      prevWasDelta = false;
    }

    // 4. Russian Roulette (Path Termination)
    float p = max(throughput.r, max(throughput.g, throughput.b));
    if (RandomValue(rngState) > p) {
      break;
    }
    throughput *= 1.0f / p;
    // if (bounceIndex > 2) {
    //   float p = max(throughput.r, max(throughput.g, throughput.b));
    //
    //   // Clamp to prevent massive energy spikes (e.g. max multiplier is 10x)
    //   p = clamp(p, 0.1f, 1.0f); 
    //
    //   if (RandomValue(rngState) > p) {
    //     break;
    //   }
    //   throughput *= 1.0f / p;
    // }
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
  // Pitch (up/down) around the XZ plane
  pR(ro.yz, u_cameraRot.y);
  // Yaw (left/right) around the vertical axis
  pR(ro.xz, u_cameraRot.x);
}

vec2 RandomPointInCircle(inout uint state) {
    float angle = RandomValue(state) * c_twopi;
    float radius = sqrt(RandomValue(state)); // sqrt ensures uniform distribution
    return vec2(cos(angle), sin(angle)) * radius;
}

void main() {
  // RNG setup
  uint state =
    uint(uint(gl_FragCoord.x) * uint(1973) +
        uint(gl_FragCoord.y) * uint(9277) + uint(u_frame) * uint(26699)) |
      uint(1);

  state = NextRandom(state);
  state = NextRandom(state);
  vec3 accumulated_color = vec3(0.0);
  int samples_per_frame = u_spf;
  float focusDist = u_focusDist;  // How far away the perfectly in-focus plane is
  float aperture = u_aperture;  // The radius of the lens (higher = blurrier background)

  for (int i = 0; i < samples_per_frame; i++) {
    vec2 jitter = vec2(RandomValue(state), RandomValue(state)) - 0.5;
    vec2 uv = (gl_FragCoord.xy + jitter) / u_resolution.xy;
    vec2 pixelTarget2D = (uv * 2.0f) - 1.0f;
    float aspectRatio = u_resolution.x / u_resolution.y;

    // calculate the camera distance
    float cameraDistance = 1.0f / tan(c_FOVDegrees * 0.5f * c_pi / 180.0f);

    // Ray ray;
    // // 1. The origin is strictly controlled by WASD
    // ray.origin = u_cameraPos;

    // 2. Base ray direction (pointing down the -Z axis like a standard camera)
    vec3 rayDir = normalize(
        vec3(pixelTarget2D.x * aspectRatio, pixelTarget2D.y, -cameraDistance));

    // 3. Rotate the ray itself based on mouse movement
    pR(rayDir.yz, u_cameraRot.y); // Pitch up/down
    pR(rayDir.xz, u_cameraRot.x); // Yaw left/right
    
    vec3 focal_point = u_cameraPos + rayDir * focusDist;

    // 3. Calculate camera's Right and Up vectors in world space 
    // by applying the exact same rotations to base axis vectors
    vec3 camRight = vec3(1.0, 0.0, 0.0);
    pR(camRight.yz, u_cameraRot.y);
    pR(camRight.xz, u_cameraRot.x);

    vec3 camUp = vec3(0.0, 1.0, 0.0);
    pR(camUp.yz, u_cameraRot.y);
    pR(camUp.xz, u_cameraRot.x);

    // 4. Jitter the ray origin within the aperture disk
    vec2 lensJitter = RandomPointInCircle(state) * aperture;
    
    Ray ray;
    ray.origin = u_cameraPos + camRight * lensJitter.x + camUp * lensJitter.y;

    // 5. Point the jittered ray at the original focal point
    ray.direction = normalize(focal_point - ray.origin);


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
