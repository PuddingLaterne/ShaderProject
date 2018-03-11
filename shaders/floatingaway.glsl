#version 330

#define FADE

const float PI = 3.1415926535897932384626433832795;
const float TWOPI = 2.0 * PI;
const float PIHALF = PI / 2.0;
const float EPSILON = 0.00001;
const float INFINITY = 1000000000.0;

const int MOTIONBLURSAMPLES = 4;
const float MOTIONBLURDELTA = 0.01;
const int DOFSAMPLES = 4;
const float DOFDELTA = 0.01;

const float FOCUS = 4.0;

const int NUMLAYERS = 4;
const float[4] LAYERS = float[4](2.0, 4.0, 6.0, 12.0);

const vec3 ORB = vec3(0.9, 0.4, 0.0);
const vec3 GLOW = vec3(1.0, 1.0, 0.0);
const vec3 BG = vec3(0.0, 0.1, 0.1);

const vec2 FOG = vec2(8.0, 20.0);
const vec3 MOVEMENT = vec3(0.2,-0.4,0.0);

uniform vec2 iResolution;
uniform float iGlobalTime;

struct ray{ vec3 o; vec3 d; };
struct result{ float t; vec3 n; };

struct sphere{ vec3 c; float r; };
struct plane{ vec3 n; float d; };

vec3 getRayDirection(vec2 pos, vec2 res, float fov)
{
	float fx = tan(radians(fov) * 0.5) / res.x;
	vec2 d = (2 * pos - res) * fx;
	return normalize(vec3(d, 1.0));
}

result intersectPlane(ray r, plane pl)
{
	result res;
	res.t = -INFINITY;

	float d = dot(pl.n, r.d);
	if(abs(d) < EPSILON)
		return res;
	res.t = (-pl.d - dot(pl.n, r.o)) / d;
	res.n = pl.n;
	return res;
}

result intersectSphere(ray r, sphere s)
{
	result res;
	res.t = -INFINITY;
	
	vec3 v = r.o - s.c;
	float vd = dot(v, r.d);
	float d = vd * vd - (dot(v, v) - s.r * s.r);
	if(d < EPSILON)
		return res;
	float q = sqrt(d);	
	res.t = (-vd -q) > 0.0 ? -vd -q : -vd + q;
	res.n = normalize((r.o + r.d * res.t) - s.c);
	return res;
}

float rand(float seed)
{	
	return fract(sin(seed) * 1231534.9);
}

float rand(vec2 seed)
{
	return rand(dot(seed, vec2(12.9898, 783.233)));
}

vec2 rand2D(vec2 seed)
{
	float r = rand(seed) * TWOPI;
	return vec2(cos(r), sin(r));
}

float fresnel(vec3 v, vec3 n, float p)
{
	return clamp(pow(1.0 + dot(v,n),p),0.0,1.0);
}

float specular(vec3 v, vec3 n, vec3 l, float shininess)
{
	vec3 r = reflect(l, n);
	return pow(max(0.0, dot(r, v)), shininess);
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

float distSphere(vec3 p, float r)
{
	return length(p) - r;
}

float smin(float a, float b, float k)
{
	float h = clamp( 0.5+0.5*(b-a)/k, 0.0, 1.0 );
    return mix( b, a, h ) - k*h*(1.0-h);
}

float distanceField(vec3 p, float t, sphere s, float seed)
{
	p -= s.c;

	float rnd = rand(seed*100.0);
	t += rnd;
	t *= mix(2.0,8.0,rnd);
	t *= sign(rand(seed*20.0)-0.5);

	rnd = rand(seed*200.0);
	float rot = p.y-s.r*0.2;
	p = rotateX(p, PI*(rnd-0.5)*0.4 + sin(t)*rot*2.0);
	p = rotateZ(p, PI*(rnd-0.5)*0.2 + cos(t)*rot);
	p = rotateY(p, t);
	vec3 op = p;
	
	p.y *= 0.8;
	p.xz *= 1.1;
	float body = distSphere(p,s.r*0.5);
	
	p = op;
	p.y -= s.r*0.06;
	rot = pow(p.x,2.0)*10.0;
	p = rotateZ(p, rot*sin(t*2.0));
	p = rotateY(p, rot*cos(t*2.0));
	p.x *= 0.8;
	p.yz *= 3.0;
	float arms = distSphere(p,s.r*0.5);
	
	p = op;
	p.y -= s.r*0.2;
	p.z += s.r*0.4;
	p.x *= 2.0;
	float mouth = distSphere(p,s.r*0.1);
	
	p = op;
	p.yz += s.r*vec2(-0.3,0.3);
	vec3 o = s.r*vec3(0.2,0.0,0.0);
	float eyes = min(distSphere(p+o,s.r*0.1), distSphere(p-o,s.r*0.1));
	
	float d = smin(body,arms,0.008);
	d = max(d, -mouth);
	d = max(d, -eyes);
	return d;
}

vec3 getNormal(vec3 p, float t, sphere s, float rnd)
{
	vec2 d = vec2(0.01, 0.0);
	float dx = distanceField(p + d.xyy,t,s,rnd)
				- distanceField(p - d.xyy,t,s,rnd);
	float dy = distanceField(p + d.yxy,t,s,rnd)
				- distanceField(p - d.yxy,t,s,rnd);
	float dz = distanceField(p + d.yyx,t,s,rnd)
				- distanceField(p - d.yyx,t,s,rnd);
	return normalize(vec3(dx, dy, dz));
}

float getInsideGlow(ray r, vec3 p, float t, sphere s, float rnd)
{
	vec3 n = getNormal(p,t,s,rnd);	
	return fresnel(r.d,n,2.0);
}

float inside(ray r, float time, sphere s, float rnd)
{		
	float t = 0.0;
	vec3 p;	
	while(t <= s.r)
	{
		p = r.o + r.d * t;
		float d = distanceField(p, time, s,rnd);		
		if(d <= EPSILON) 
		{
			float glow = getInsideGlow(r,p,time,s,rnd);
			glow *= 1.0-t/(s.r*2.0);
			return glow;
		}
		t += max(d, t * 0.0001);
	}	
	return 0.0;
}

vec4 getColor(ray r, result res, float t, float rnd, sphere s)
{
	if(res.t < 0)return vec4(0.0);
	
	r.o = r.o + res.t * r.d;
				
	vec3 lPos = MOVEMENT*t + vec3(0.0, 2.0, 2.0);
	vec3 l = normalize(lPos-r.o);
	
	float glow = inside(r,iGlobalTime,s,rnd);
	glow *= smoothstep(0.4, 1.0, rand(rnd*40.0));
	glow *= 2.0;
	glow += (sin((t+TWOPI*rnd)*3.0)+1.0)*0.2;
	glow += specular(r.d, res.n, l, 16.0)*0.4;
	
	vec3 color = ORB + GLOW * glow;
	
	float a = 1.0;
	a *= pow(dot(r.d,-res.n),1.5);
	a *= rnd;	
	a *=  1.0 - smoothstep(FOG.x,FOG.y,res.t);
	return vec4(color,a);
}


vec3 spheres(ray r, float t, vec3 bg)
{
	r.o += t*MOVEMENT;
	vec4 color = vec4(bg,0.0);
	for(int i = 0; i < NUMLAYERS; i++)
	{
		plane pl = plane(vec3(0.0,0.0,-1.0),LAYERS[i]);
		vec3 p = r.o + intersectPlane(r,pl).t * r.d;
		p = vec3(floor(p.xy)+vec2(0.5),p.z);
		
		float rnd = rand(p.xy+float(i));
		
		float radius = mix(0.01, 0.3, rnd);	
		vec2 offset = rand2D(p.xy+i);
		offset.x *= sin(3.0*radius*t+rnd);
		p.xy += offset*(0.5-radius);
		
		sphere s = sphere(p, radius);	
		vec4 c = getColor(r,intersectSphere(r, s),t,rnd,s);

		float a = c.a*(1.0-color.a);
		color.rgb = mix(color.rgb, c.rgb, a);
		color.a += a;
	}	
	return color.rgb;
}

vec3 pow3D(vec3 x, float p)
{
	return vec3(pow(x.x,p),pow(x.y,p),pow(x.z,p));
}

float stars(vec2 uv, float amount, float radius)
{
	uv = uv * amount;
	vec2 gridID = floor(uv);
	vec2 starPos = vec2(rand(gridID),rand(gridID+1));
	starPos = (starPos - 0.5) * 2.0;
	starPos = vec2(0.5) + starPos * (0.5 - radius * 2.0);
	float stars = distance(fract(uv), starPos);
	float size = rand(gridID)*radius;
	stars = 1.0 - smoothstep(0.0, size + radius, stars);
	return stars;
}

vec3 background(vec2 uv)
{
	float f = 1.0-(uv.y+1.0)/2.0;
	f += stars(uv, 4.0, 0.06)*0.3;
	vec3 bg = mix(pow3D(BG,4.0), BG, f/1.3);
	return bg;
}

const vec2 FADE_T = vec2(4.0,15.0);
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
		coord.y -= iResolution.y * (1.0 - fadeIn());
		coord.y -= iResolution.y * fadeOut() * 2.0;
	#endif

	vec2 uv = vec2(2.0*coord - iResolution)/iResolution.y;
	ray r;
	r.o = vec3(0.0);
	r.d = getRayDirection(coord,iResolution.xy, 60.0);
		
	vec3 fp = r.o + intersectPlane(r,plane(vec3(0.0,0.0,-1.0),FOCUS)).t*r.d;
	
	vec3 bg = background(uv);
	vec3 color = vec3(0.0);
	float total;
	for(int i = 0; i < MOTIONBLURSAMPLES; i++)
	{
		float t = iGlobalTime - MOTIONBLURDELTA*i;
		float falloff = pow(0.8, i);
		total += falloff;	
		vec3 c = spheres(r,t,bg);
		for(int j = 0; j < DOFSAMPLES; j++)
		{
			ray r;
			r.o.xy = rand2D(vec2(j,rand(j)))*DOFDELTA;
			r.d = normalize(fp - r.o);
			c += spheres(r,t,bg);
		}
		color += c*falloff;
	}
	color /= total*float(DOFSAMPLES+1);
	#ifdef FADE
		color = fade(color);
	#endif
	gl_FragColor = vec4(color,1.0);
}