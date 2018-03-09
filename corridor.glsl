#version 330

//#define SHOWHEIGHTFIELD
#define FADE

uniform vec2 iResolution;
uniform float iGlobalTime;
uniform sampler2D texLastFrame1;

out vec4 image;
out vec4 buffer;

const float PI = 3.1415926535897932384626433832795;
const float TWOPI = 2.0 * PI;
const float PIHALF = PI / 2.0;
const float EPSILON = 0.00001;
const float INFINITY = 1000000000;

const float drawDist = 60.0;
const float thickness = 0.2;
const float lampSpacing = 4.0;
const vec3 lPos = vec3(lampSpacing, 3.0, -2.0);

const vec3 cStonesA = vec3(0.9, 1.0, 0.8);
const vec3 cStonesB = cStonesA * vec3(1.4, 1.3, 1.4);
const vec3 cVines = vec3(0.3, 0.5, 0.1);
const vec3 cLamps = vec3(2.0, 2.0, 2.0);
const vec3 light = vec3(1.1, 1.1, 1.0);
const vec3 shadow = vec3(0.0, 0.08, 0.06);
const vec3 bg = cStonesB * shadow;

float smootherstep(float a, float b, float x)
{
    x = clamp((x - a)/(b - a), 0.0, 1.0);
    return x*x*x*(x*(x*6.0 - 15.0) + 10.0);
}

vec2 smootherstep(float a, float b, vec2 i)
{
	return vec2(smootherstep(a, b, i.x), smootherstep(a, b, i.y));
}

float rand(float seed)
{	
	return fract(sin(seed) * 1231534.9);
}

float rand(vec2 seed)
{
	return rand(dot(seed, vec2(12.9898, 783.233)));
}

float rand(vec2 seed, float f)
{
	seed = mod(seed, f);
	return rand(seed);
}

vec2 rand2D(vec2 seed, float f)
{
	float r = rand(seed, f) * TWOPI;
	return vec2(cos(r), sin(r));
}

vec2 rand2D01(vec2 seed, float s)
{
	vec2 r = rand2D(seed, s);
	return (r + vec2(1.0))/2.0;
}

float voronoi(vec2 x, float scale)
{
	x *= scale;
	vec2 i = floor(x);
    vec2 f = fract(x);

	vec2 mn; //closest neighbour
	vec2 mc; //closest center
    float dc = INFINITY; //distance to center
    for( int x=-1; x<=1; x++ )
    for( int y=-1; y<=1; y++ )
    {
        vec2 n = vec2(x,y);
        vec2 c = n + rand2D01(i+n,scale) - f;
        float d = dot(c,c);
        if( d<dc ){ dc = d; mc = c; mn = n;}
    }
    float db = INFINITY; //distance to border
    for( int x=-2; x<=2; x++ )
    for( int y=-2; y<=2; y++ )
    {
        vec2 n = mn + vec2(x,y);
        vec2 c = n + rand2D01(i+n,scale) - f;
        float d = dot(0.5*(mc+c), normalize(c-mc));
        db = min(db,d);
    }
	return 1.0 - mix(db, 0.5-dc, db*2.0)*2.0;
}

float gradientNoise(vec2 coord, float frequency)
{
	vec2 i = coord * frequency;
	vec2 f = fract(i);	
	i = floor(i);
	
	vec2 g00 = rand2D(i,frequency);
	vec2 g01 = rand2D(i + vec2(0.0, 1.0),frequency);
	vec2 g10 = rand2D(i + vec2(1.0, 0.0),frequency);
	vec2 g11 = rand2D(i + vec2(1.0, 1.0),frequency);
	
	float v00 = dot(g00, f);
	float v10 = dot(g10, f - vec2(1.0, 0.0));
	float v01 = dot(g01, f - vec2(0.0, 1.0));
	float v11 = dot(g11, f - vec2(1.0, 1.0));

	f = smootherstep(0.0, 1.0, f);
	float x1 = mix(v00, v10, f.x);
	float x2 = mix(v01, v11, f.x);
	
	float n =  mix(x1, x2, f.y);	
	return n;	
}

float fbm(vec2 coord, float frequency)
{
	int octaves = 4;
	float lacunarity = 2.0;
	float gain = 0.8;

	float amplitude = 0.5;
	
	float n = 0.5;
	
	for(int i = 0; i < octaves; i++)
	{
		n += gradientNoise(coord, frequency) * amplitude;
		frequency *= lacunarity;
		amplitude *= gain;
	}
	return n;	
}

vec2 rep(vec2 p, vec2 r)
{
	return mod(p,r)-r/2.0;
}

float smin(float a, float b, float k)
{
	float h = clamp( 0.5+0.5*(b-a)/k, 0.0, 1.0 );
    return mix( b, a, h ) - k*h*(1.0-h);
}

float sdCircle(vec2 p, float r)
{
	return length(p) - r;
}

float sdCapsule(vec2 p, float r, float h)
{
	p.y -= smoothstep(0.0, h, abs(p.y))*h*sign(p.y);
	return sdCircle(p, r);
} 

vec2 rotate(vec2 p, float angle)
{
	mat2 mat = mat2(cos(angle),-sin(angle),
					sin(angle),cos(angle));
	return mat * p;
}

float dancers(vec2 p, float t, inout vec3 c)
{		
	float cutoff = 1.0 - step(1.0, abs(p.y));
	float f = ((p.y+0.6)/1.2)*sin(t); 
	p.x += cos(t)*0.2;
	p = rep(p, vec2(1.5, 0.0));
	p = rotate(p, f);
	vec2 op = p;
	p.x *= mix(1.0, 0.8, abs(f));
	float body = sdCapsule(p, 0.2, 0.4);
	
	float eyes = 1.0 - smoothstep(0.02, 0.05, length(p+vec2(0.1, -0.46)));
	eyes += 1.0 - smoothstep(0.02, 0.05, length(p+vec2(-0.1, -0.46)));
	float mouth = sdCircle((p+vec2(0.0, -0.4))*vec2(1.0, 0.6+cos(t*2.0)*0.2), 0.03);
	
	p = op;
	p.y -= 0.1;
	p = rotate(p, sin(t*2.0)*0.5 * smoothstep(0.2, 0.6, abs(p.x)) + PIHALF);
	float arms = sdCapsule(p, 0.05, 0.4);
	
	p = op;
	p.y += 0.4;
	p = rotate(p, PIHALF - p.x*3.0);
	float legs = sdCapsule(p, 0.1, 0.3);

	body = max(body, -mouth);
	body = smin(body, arms, 0.1);
	body = smin(body, legs, 0.2);
	body = 1.0 - smoothstep(-0.02, 0.02, body);
	
	float dancers = (body + eyes)*cutoff;
	c = mix(c, cLamps, eyes*cutoff*0.4);
	return dancers * 0.1;
}

vec2 spiralCoords(vec2 p, float turns) 
{
	p = vec2(atan(p.x, p.y)/TWOPI + 0.5, length(p)*turns);
    float s = p.y+p.x;
    float l = (floor(s)-p.x);
    float d = fract(s);
    return vec2(l, d);
}

float spiral(vec2 p)
{
	float t = 0.05;
	float s = 0.3;
	p = spiralCoords(p,2.0);
	float d = smoothstep(0.5-t-s, 0.5-t, p.y);
	d *= 1.0 - smoothstep(0.5+t, 0.5+t+s, p.y);
	d *= 1.0 - smoothstep(1.0,1.1, p.x);
	return 1.0 - d;
}

float ornament(vec2 p)
{
	vec2 op = p;
	
	//swirls
	p += vec2(1.0, -0.1);
	float d = spiral(p);
	p += vec2(-2.0, 0.2);
	p *= -1.0;
	d = min(d, spiral(p));

	//connection piece
	p = op;
	float c = abs(p.y - mix(-0.65, 0.65, smootherstep(-1.0, 1.0, p.x)));
	c = smoothstep(0.00, 0.15, c);
	c += smoothstep(1.0, 1.1, abs(p.x));
		
	d = min(d, c);
	return 1.0 - d;
}

float border(vec2 p, float t, inout vec3 c)
{
	p.y = abs(p.y);
	p.y -= 1.4;	
	vec2 op = p;
	
	p *= 3.0;
	p = rep(p, vec2(4.0, 0.0));
	float ornament = ornament(p);
	p.x = abs(p.x);
	float points = 1.0 - smoothstep(0.0, 0.15, distance(p,vec2(2.0,0.0)));	
	
	p = op;
	p.y = abs(p.y);
	float lines = 1.0 - smoothstep(0.02,0.1, abs(p.y - 0.5));
	float border = 1.0 - smoothstep(0.1, 0.6, p.y);

	return ornament*0.1 + lines * 0.2 + points * 0.1 + border * 0.9;
}

float pnorm(vec2 v, float p)
{
	return pow(pow(v.x,p)+pow(v.y,p),1.0/p);
}

float lamps(vec2 p, float t)
{
	p.y -= lPos.y;
	p.x -= lampSpacing;
	p = rep(p, vec2(lampSpacing*2.0, 0.0));
	return smoothstep(0.0, 1.0, length(p));
}

float stones(vec2 p, float t, inout vec3 c)
{
	vec2 op = p;
	float n = texture(texLastFrame1, op*0.6).x;
	
	p.x *= 0.4;
	p.x += fract(floor(p.y)/2.0);
	float r = rand(floor(p));
	p = fract(p);
	p -= vec2(0.5);
	p = abs(p);
	float h = 1.0 - smoothstep(0.3,0.52,pnorm(p,mix(3.0, 6.0, r)));
	h *= mix(0.6, 1.0, r);
	//clean up in the middle
	h *= step(2.0, abs(op.y));
	h += n*0.05;
	h -= smoothstep(0.5, 1.0, n)*0.06;
	
	float lamps = lamps(op, t);
	h *= smoothstep(0.2,0.4, lamps);
	lamps= 1.0 - smoothstep(0.0, 0.4, lamps);
	h += lamps;
	c = mix(cStonesA, cStonesB, n);
	c = mix(c, cLamps, lamps);
	return h*0.9;
}

float vines(vec2 p, float t, inout vec3 c)
{
	vec2 bS = texture(texLastFrame1, p*0.4).xy;
	vec2 bL = texture(texLastFrame1, p*0.08).xy;
	float h = (bS.y + (bS.x-0.5)*0.8)/1.4;
	h = smoothstep(0.0,1.0, h);
	h /= 2.0;
	h *= smoothstep(0.55, 0.65, bL.x);
	h *= smoothstep(1.0, 2.0, abs(p.y));
	c = mix(c, cVines, h);
	return h*0.1;
}

float heightField(vec2 p, float t, inout vec3 c)
{
	float h = stones(p,t, c);
	h += border(p, t, c);
	h += dancers(p,t, c);
	h += vines(p,t, c);
	return h;
}

float heightField(vec2 p, float t)
{
	vec3 c;
	return heightField(p, t, c);
}

vec4 sampleNeighbours(vec2 uv, float t)
{
	vec2 d = vec2(0.01, 0.0);
	vec4 s;
	s.x = heightField(uv - d.xy, t);
	s.y = heightField(uv + d.xy, t);
	s.z = heightField(uv - d.yx, t);
	s.w = heightField(uv + d.yx, t);
	return s;
}

vec3 getNormal(vec4 s)
{	
	vec2 d = vec2(0.01, 0.0);
	float dx = s.x - s.y;
	float dy = s.z - s.w;
	
	vec3 x = vec3(2.0 * d.x, dx * 0.0, dx*thickness);
	vec3 y =  vec3(0.0, d.x * 2.0, dy*thickness);
	 
	return normalize(cross(y, x));	
}

float lambert(vec3 p, vec3 n, vec3 lPos)
{
	vec3 l = lPos - p;
	float i = max(0.0, dot(normalize(l), n));
	l.y *= 0.6;
	i *= 1.0 - pow(length(l/(lampSpacing*1.5)),2.0);
	return clamp(i,0,1);
}

float ambientOcclusion(float h, vec4 s)
{
	float ao = max(s.x - h,0.0);
	ao += max(s.y - h,0.0);
	ao += max(s.z - h,0.0);
	ao += max(s.w - h,0.0);
	return 1.0 - smoothstep(0.0, 0.2, ao);
}

vec3 rep(vec3 p, vec3 r)
{
	return mod(p, r)-0.5*r;
}

float intensity(vec3 p, float h, float t)
{
	vec4 s = sampleNeighbours(p.xy, t);
	vec3 n = getNormal(s);				
	vec3 lp = rep(p, vec3(lampSpacing*2.0, 0.0, 0.0));
	float i = lambert(lp, n, lPos);
	i += lambert(lp, n, lPos * vec3(-1.0, 1.0, 1.0));
	i = mix(0.05, 1.0, i);
	i *= ambientOcclusion(h,s);
	return i;
}

vec3 rotateX(vec3 p, float angle)
{
	mat3 r = mat3(1, 0, 0,
					0, cos(angle), -sin(angle),
					0, sin(angle), cos(angle));
	return r * p;
}

vec3 rotateY(vec3 p, float angle)
{
	mat3 r = mat3(cos(angle), 0, sin(angle),
				0, 1, 0,
				-sin(angle),0 , cos(angle));
	return r * p;
}

vec3 rotateZ(vec3 p, float angle)
{
	mat3 r = mat3(cos(angle), -sin(angle), 0,
				sin(angle), cos(angle), 0,
				0, 0, 1);
	return r * p;
}

struct ray
{
	vec3 o;
	vec3 d;
};

ray createRay(float fov, vec2 coord, vec2 res)
{
	ray r;
	r.o = vec3(iGlobalTime, 0.0, -5.0);
	float fx = tan(radians(fov) / 2.0) / res.x;
	vec2 d = fx * ( coord.xy * 2.0 - res);
	r.d = normalize(vec3(d, 1.0));
	r.d = rotateX(r.d, 0.1 + 0.2*sin(iGlobalTime));
	r.d = rotateY(r.d, -0.7 + 0.16*cos(iGlobalTime));
	return r;
}

struct plane
{
	vec3 n;
	float d;
};
plane frontPlane = plane(vec3(0.0, 0.0, -1.0), 0.0);	
plane backPlane = plane(vec3(0.0, 0.0, -1.0), thickness);	

float intersectPlane(ray r, plane pl)
{
	float d = dot(pl.n, r.d);
	if(abs(d) < EPSILON)
		return -INFINITY;;
	return (-pl.d - dot(pl.n, r.o)) / d;
}

float map01(float min, float max, float t)
{
	return clamp((t - min )/ (max - min),0,1);
}

float vignette(vec2 uv)
{
	float v = pnorm((uv-vec2(0.5))*2.0,6.0);
	v = smoothstep(0.9, 1.2, v);
	return v;
}	

vec3 corridor(ray r, vec2 uv)
{
	float time = iGlobalTime * 3.0;
		
	float front = intersectPlane(r, frontPlane);
	float back = intersectPlane(r, backPlane);
	
	vec3 color = bg;	
	if(front > 0.0 && back > 0.0 && back < drawDist)
	{	
		vec3 p = r.o + front * r.d;
		float h = 0.0;
		while(p.z < thickness)
		{
			float ph = 1.0 - map01(0, thickness, p.z);
			h = heightField(p.xy, time, color);
			if(ph <= h) break;
			p += r.d * 0.01;
		}
		float i = intensity(p, h, time);
		color = mix(color * shadow, color * light, i);	
	}
	float fog = map01(drawDist/10.0, drawDist, back);
	return mix(color, bg, max(fog,vignette(uv)));
}

vec4 textures(vec2 uv)
{
	float fbm = fbm(uv,12.0);
	float voronoi = voronoi(uv, 12.0);
	return vec4(fbm, voronoi, 0.0, 0.0);
}

const vec2 FADE_T = vec2(18.0,32.0);
const vec2 FADE_D = vec2(1.0, 1.0);

float fadeIn(){return smoothstep(FADE_T.x, FADE_T.x+FADE_D.x,iGlobalTime);}
float fadeOut(){return smoothstep(FADE_T.y-FADE_D.y, FADE_T.y,iGlobalTime);}

vec3 fade(vec3 color)
{
	vec3 from = vec3(0.0);
	vec3 to = vec3(0.0);
	color = mix(from, color, fadeIn());
	color = mix(color, to, fadeOut());
	return color;
}

void main()
{
	vec2 coord = gl_FragCoord.xy;
	#ifdef FADE
		coord.y += iResolution.y * (1.0 - fadeIn());
		coord.y -= iResolution.y * fadeOut() * 2.0;
	#endif

	vec2 uv = gl_FragCoord.xy / iResolution.xy;
	
	vec3 color;
	#ifdef SHOWHEIGHTFIELD
		color = vec3(1.0) * heightField((uv-vec2(0.5))*10.0,iGlobalTime*3.0);
	#else
		ray r = createRay(70.0, coord, iResolution.xy);		
		color = corridor(r,uv);	
	#endif	
	
	#ifdef FADE
		color = fade(color);
	#endif	
	image = vec4(color,1.0);
	buffer = textures(uv);
}