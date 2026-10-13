const BASE = process.env.BACKEND_URL || "http://127.0.0.1:8091";

async function proxy(request: Request, path: string[]) {
  const incoming = new URL(request.url);
  const target = `${BASE}/${path.join("/")}${incoming.search}`;
  const headers = new Headers();
  const contentType = request.headers.get("content-type");
  if (contentType) headers.set("content-type", contentType);
  const init: RequestInit = { method: request.method, headers, cache: "no-store" };
  if (request.method !== "GET" && request.method !== "HEAD") {
    init.body = await request.text();
  }
  try {
    const response = await fetch(target, init);
    const body = await response.arrayBuffer();
    return new Response(body, {
      status: response.status,
      headers: {
        "content-type": response.headers.get("content-type") || "application/json",
      },
    });
  } catch {
    return Response.json(
      {
        detail:
          "The model service is not running. From the repo root, start it with: uvicorn backend.app.main:app --host 127.0.0.1 --port 8091",
      },
      { status: 503 },
    );
  }
}

type RouteContext = { params: Promise<{ path: string[] }> };

export async function GET(request: Request, context: RouteContext) {
  const { path } = await context.params;
  return proxy(request, path);
}

export async function POST(request: Request, context: RouteContext) {
  const { path } = await context.params;
  return proxy(request, path);
}
