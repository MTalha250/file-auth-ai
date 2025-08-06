"use client";
import { MessageSquare } from "lucide-react";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import api from "../../lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  FileText,
  Upload,
  LogOut,
  Check,
  AlertCircle,
  Calendar,
  User,
  Download,
  Eye,
  Trash2,
  Loader2,
  FileCheck,
  Database,
} from "lucide-react";

type Category = {
  id: number;
  category_name: string;
  description: string;
  expected_fields: string[];
  created_at: string;
  updated_at: string;
};

export default function DashboardPage() {
  const [categories, setCategories] = useState<Category[]>([]);
  const [file, setFile] = useState<File | null>(null);
  const [category, setCategory] = useState("");
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [dragActive, setDragActive] = useState(false);
  const [chatQuestion, setChatQuestion] = useState("");
  const [chatHistory, setChatHistory] = useState<Array<{question: string, answer: string}>>([]);
const [chatLoading, setChatLoading] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState<Category | null>(
    null
  );
  const router = useRouter();

  useEffect(() => {
    api
      .get<Category[]>("/api/v1/categories/")
      .then((res) => setCategories(res.data))
      .catch(() => setError("Failed to load categories"));
  }, []);

  useEffect(() => {
    const selected = categories.find((cat) => cat.category_name === category);
    setSelectedCategory(selected || null);
  }, [category, categories]);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;

    setLoading(true);
    setResult(null);
    setError("");

    const formData = new FormData();
    formData.append("category", category);
    formData.append("file", file);
    formData.append("file_name", file.name);

    try {
      const res = await api.post("/api/v1/submit-file/", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResult(res.data);
    } catch (err: any) {
      setError(
        err?.response?.data?.error || "Upload failed. Please try again."
      );
    } finally {
      setLoading(false);
    }
  };

  const handleAskQuestion = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!chatQuestion.trim()) return;
    
    setChatLoading(true);
    try {
      const response = await api.post("/api/v1/ask-rag-question/", {
        question: chatQuestion
      });
      
      setChatHistory(prev => [...prev, {
        question: chatQuestion,
        answer: response.data.answer
      }]);
      setChatQuestion("");
    } catch (error: any) {
      setError(error?.response?.data?.error || "Failed to get answer");
    } finally {
      setChatLoading(false);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem("access");
    localStorage.removeItem("refresh");
    router.push("/login");
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  };

  const getFileIcon = (fileName: string) => {
    const extension = fileName.split(".").pop()?.toLowerCase();
    switch (extension) {
      case "pdf":
        return <FileText className="w-8 h-8 text-red-500" />;
      case "doc":
      case "docx":
        return <FileText className="w-8 h-8 text-blue-500" />;
      case "xls":
      case "xlsx":
        return <Database className="w-8 h-8 text-green-500" />;
      default:
        return <FileText className="w-8 h-8 text-gray-500" />;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100">
      {/* Background decorative elements */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-blue-400 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-pulse"></div>
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-indigo-400 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-pulse delay-1000"></div>
      </div>

      {/* Header */}
      <div className="relative bg-white/80 backdrop-blur-sm border-b border-slate-200/50 shadow-sm">
        <div className="max-w-6xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-xl flex items-center justify-center">
                <FileText className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                  FileAuthAI Dashboard
                </h1>
                <p className="text-sm text-slate-600">
                  Document Processing & Analysis
                </p>
              </div>
            </div>
            <Button
              variant="outline"
              onClick={handleLogout}
              className="flex items-center space-x-2 hover:bg-red-50 hover:border-red-200 hover:text-red-600 transition-colors"
            >
              <LogOut className="w-4 h-4" />
              <span>Logout</span>
            </Button>
          </div>
        </div>
      </div>

      <div className="relative max-w-6xl mx-auto px-4 py-8">
        <div className="grid lg:grid-cols-3 gap-8">
          {/* Main Upload Form */}
          <div className="lg:col-span-2 space-y-6">
            <Card className="backdrop-blur-sm bg-white/80 shadow-xl border-0 ring-1 ring-slate-200/50">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2 text-slate-800">
                  <Upload className="w-5 h-5" />
                  <span>Submit Document</span>
                </CardTitle>
                <p className="text-sm text-slate-600">
                  Upload and process your documents with AI analysis
                </p>
              </CardHeader>

              <CardContent className="space-y-6">
                <div className="space-y-4">
                  {/* Category Selection */}
                  <div className="space-y-2">
                    <label
                      htmlFor="category"
                      className="text-sm font-medium text-slate-700 flex items-center space-x-2"
                    >
                      <Database className="w-4 h-4" />
                      <span>Document Category</span>
                    </label>
                    <select
                      id="category"
                      className="w-full appearance-none h-11 px-3 border border-slate-200 rounded-lg bg-white focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 transition-all duration-200"
                      value={category}
                      onChange={(e) => setCategory(e.target.value)}
                      required
                    >
                      <option value="">Select a category</option>
                      {categories.map((cat) => (
                        <option key={cat.id} value={cat.category_name}>
                          {cat.category_name}
                        </option>
                      ))}
                    </select>
                  </div>

                  {/* File Upload Area */}
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-slate-700 flex items-center space-x-2">
                      <FileText className="w-4 h-4" />
                      <span>Document File</span>
                    </label>

                    <div
                      className={`relative border-2 border-dashed rounded-lg p-8 text-center transition-all duration-200 ${
                        dragActive
                          ? "border-blue-400 bg-blue-50"
                          : "border-slate-300 hover:border-slate-400"
                      }`}
                      onDragEnter={handleDrag}
                      onDragLeave={handleDrag}
                      onDragOver={handleDrag}
                      onDrop={handleDrop}
                    >
                      <input
                        type="file"
                        onChange={handleFileChange}
                        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                        required
                      />

                      {file ? (
                        <div className="flex items-center justify-center space-x-3">
                          {getFileIcon(file.name)}
                          <div className="text-left">
                            <p className="text-sm font-medium text-slate-700">
                              {file.name}
                            </p>
                            <p className="text-xs text-slate-500">
                              {formatFileSize(file.size)}
                            </p>
                          </div>
                          <Button
                            type="button"
                            variant="outline"
                            size="sm"
                            onClick={() => setFile(null)}
                            className="text-red-600 hover:text-red-700"
                          >
                            <Trash2 className="w-4 h-4" />
                          </Button>
                        </div>
                      ) : (
                        <div className="space-y-3">
                          <Upload className="w-12 h-12 text-slate-400 mx-auto" />
                          <div>
                            <p className="text-sm font-medium text-slate-700">
                              Drop your file here, or click to browse
                            </p>
                            <p className="text-xs text-slate-500 mt-1">
                              Supports PDF, DOC, DOCX, XLS, XLSX files
                            </p>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>

                  {error && (
                    <Alert
                      variant="destructive"
                      className="bg-red-50 border-red-200"
                    >
                      <AlertCircle className="w-4 h-4" />
                      <AlertDescription>{error}</AlertDescription>
                    </Alert>
                  )}

                  <Button
                    type="submit"
                    onClick={handleSubmit}
                    className="w-full h-11 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white font-medium rounded-lg shadow-lg hover:shadow-xl transform hover:scale-[1.02] transition-all duration-200"
                    disabled={loading || !file || !category}
                  >
                    {loading ? (
                      <div className="flex items-center space-x-2">
                        <Loader2 className="w-4 h-4 animate-spin" />
                        <span>Processing document...</span>
                      </div>
                    ) : (
                      <div className="flex items-center space-x-2">
                        <FileCheck className="w-4 h-4" />
                        <span>Submit Document</span>
                      </div>
                    )}
                  </Button>
                </div>
              </CardContent>
            </Card>

            {/* Results Section */}
            {result && (
              <Card className="backdrop-blur-sm bg-white/80 shadow-xl border-0 ring-1 ring-slate-200/50">
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2 text-green-600">
                    <Check className="w-5 h-5" />
                    <span>Processing Results</span>
                  </CardTitle>
                  <p className="text-sm text-slate-600">
                    AI analysis completed successfully
                  </p>
                </CardHeader>

                <CardContent>
                  <div className="bg-slate-50 rounded-lg p-4 border">
                    <div className="flex items-center justify-between mb-3">
                      <span className="text-sm font-medium text-slate-700">
                        Analysis Output
                      </span>
                      <Button variant="outline" size="sm" className="text-xs">
                        <Download className="w-3 h-3 mr-1" />
                        Export
                      </Button>
                    </div>
                    <pre className="text-xs text-slate-600 overflow-x-auto whitespace-pre-wrap bg-white p-3 rounded border">
                      {JSON.stringify(result, null, 2)}
                    </pre>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Chat Interface */}
            <Card className="backdrop-blur-sm bg-white/80 shadow-xl border-0 ring-1 ring-slate-200/50">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2 text-slate-800">
                  <MessageSquare className="w-5 h-5" />
                  <span>Ask Questions About Your Documents</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                {/* Chat messages display */}
                <div className="space-y-4 mb-4 max-h-96 overflow-y-auto">
                  {chatHistory.map((msg, index) => (
                    <div key={index} className="space-y-2">
                      <div className="bg-blue-50 p-3 rounded-lg">
                        <strong>Q:</strong> {msg.question}
                      </div>
                      <div className="bg-green-50 p-3 rounded-lg">
                        <strong>A:</strong> {msg.answer}
                      </div>
                    </div>
                  ))}
                </div>
                
                {/* Question input */}
                <form onSubmit={handleAskQuestion} className="flex space-x-2">
                  <input
                    type="text"
                    value={chatQuestion}
                    onChange={(e) => setChatQuestion(e.target.value)}
                    placeholder="Ask a question about your documents..."
                    className="flex-1 p-2 border rounded-lg"
                    disabled={chatLoading}
                  />
                  <Button type="submit" disabled={chatLoading}>
                    {chatLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : "Ask"}
                  </Button>
                </form>
              </CardContent>
            </Card>
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* Category Info */}
            {selectedCategory && (
              <Card className="backdrop-blur-sm bg-white/80 shadow-xl border-0 ring-1 ring-slate-200/50">
                <CardHeader>
                  <CardTitle className="text-lg text-slate-800">
                    Category Details
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <h4 className="font-medium text-slate-700 mb-2">
                      {selectedCategory.category_name}
                    </h4>
                    <p className="text-sm text-slate-600 mb-3">
                      {selectedCategory.description}
                    </p>
                  </div>

                  <div>
                    <h5 className="text-sm font-medium text-slate-700 mb-2">
                      Expected Fields:
                    </h5>
                    <div className="space-y-1">
                      {selectedCategory.expected_fields.map((field, index) => (
                        <div
                          key={index}
                          className="flex items-center space-x-2 text-xs text-slate-600"
                        >
                          <div className="w-1.5 h-1.5 bg-blue-400 rounded-full"></div>
                          <span>{field}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="pt-3 border-t border-slate-100">
                    <div className="flex items-center space-x-2 text-xs text-slate-500">
                      <Calendar className="w-3 h-3" />
                      <span>
                        Updated:{" "}
                        {new Date(
                          selectedCategory.updated_at
                        ).toLocaleDateString()}
                      </span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Quick Stats */}
            <Card className="backdrop-blur-sm bg-white/80 shadow-xl border-0 ring-1 ring-slate-200/50">
              <CardHeader>
                <CardTitle className="text-lg text-slate-800">
                  Quick Stats
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-slate-600">
                    Available Categories
                  </span>
                  <span className="font-medium text-slate-800">
                    {categories.length}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-slate-600">
                    Current Session
                  </span>
                  <span className="font-medium text-green-600">Active</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-slate-600">Status</span>
                  <div className="flex items-center space-x-1">
                    <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                    <span className="text-sm font-medium text-green-600">
                      Online
                    </span>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Help Section */}
            <Card className="backdrop-blur-sm bg-white/80 shadow-xl border-0 ring-1 ring-slate-200/50">
              <CardHeader>
                <CardTitle className="text-lg text-slate-800">
                  Need Help?
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <p className="text-sm text-slate-600">
                  Having trouble with document processing? Check our guidelines:
                </p>
                <div className="space-y-2 text-xs text-slate-600">
                  <div className="flex items-start space-x-2">
                    <div className="w-1 h-1 bg-blue-400 rounded-full mt-2"></div>
                    <span>Ensure files are under 10MB</span>
                  </div>
                  <div className="flex items-start space-x-2">
                    <div className="w-1 h-1 bg-blue-400 rounded-full mt-2"></div>
                    <span>Use clear, readable documents</span>
                  </div>
                  <div className="flex items-start space-x-2">
                    <div className="w-1 h-1 bg-blue-400 rounded-full mt-2"></div>
                    <span>Select the correct category</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}
