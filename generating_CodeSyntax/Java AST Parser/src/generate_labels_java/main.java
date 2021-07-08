package generate_labels_java;

import java.util.HashMap;
import java.util.Map;
import java.util.HashSet;
import java.util.Set;
import org.eclipse.jdt.core.dom.*;
import java.io.*;
import java.nio.charset.*;
import java.util.*;

public class main {
	public static String m_code;
	public static void main(String args[]){
		String labels = "";
		boolean append = false;
		for (int i =0; i<13711; i ++) { 
			if (i%100 == 0)
				System.out.println("processing code sample " + i);
			// read code
			String text="";
			try {
				Scanner scanner = new Scanner( new File("data/"+i+".txt") , "UTF-8");
				while (scanner.hasNextLine()) {
					if (text.isEmpty()) {
						text = scanner.nextLine();
					}  else {
						text += "\n"+ scanner.nextLine();
					}
				}
				scanner.close(); 
			}
			catch (Exception e) {
				System.out.println(e);
			}
			m_code = "public class A {\n" + text + "\n}";
			
			// parse AST 
			ASTParser parser = ASTParser.newParser(AST.JLS3);
			parser.setSource((m_code).toCharArray());
			parser.setKind(ASTParser.K_COMPILATION_UNIT);
			final CompilationUnit cu = (CompilationUnit) parser.createAST(null);
			
			// generate labels
			// instead of writing the index of token, we write [start position of dep, end position of dep, start position of head, 
			// end position of head]
			// Then we convert it into index of tokens with cubert java tokenizer (javalang)
			// For dep, we only use the first token within the range.
			// For head, we get index of first token and last token within the range.
			
			ArrayList<String> results = new ArrayList<String>();

			
			cu.accept(new ASTVisitor() {
				public void add_idx_tuple(String reln, int position1, int length1, int position2, int length2) {
					if (position1 < position2 && position1-17 > 0) {
						String position = (position1-17) + " " + (position1+length1-17) + " " + (position2-17) + " " + (position2+length2-17);
						results.add(reln + " " + position);
					}
				}
				
				 
				
				public boolean visit(IfStatement node) {
					
					if (!m_code.substring(node.getStartPosition(),node.getStartPosition()+2).equals("if")) {
						System.out.println("Error: 'if' is expected");
					}
					if (node.getElseStatement() != null) {
						int start_position = node.getThenStatement().getStartPosition() + node.getThenStatement().getLength();
						String code_snippet = m_code.substring(start_position, node.getElseStatement().getStartPosition());
						int else_index = -1;
						for (int i = 0; i < code_snippet.length(); i++) {
							if (code_snippet.charAt(i) == '/' && i+1 < code_snippet.length()) {
								if (code_snippet.charAt(i+1) == '/') {
									int end_of_comment = code_snippet.substring(i, code_snippet.length()).indexOf("\n");
									if (end_of_comment == -1)
										break;
									else
										i += end_of_comment;
								}
								else if (code_snippet.charAt(i+1) == '*') {
									int end_of_comment = code_snippet.substring(i, code_snippet.length()).indexOf("*/");
									if (end_of_comment == -1)
										break;
									else
										i += end_of_comment+1;
								}
							} else if (code_snippet.substring(i, i+4).equals("else")) {
								else_index = i;
								break;
							}
						}
						if (else_index != -1)
							add_idx_tuple("If:if->else", node.getStartPosition(), 2 , (start_position+else_index), 4);
					}
					
					add_idx_tuple("If:if->test", node.getStartPosition(), 2, node.getExpression().getStartPosition(), node.getExpression().getLength());
				
					
					add_idx_tuple("If:if->body", node.getStartPosition(), 2, node.getThenStatement().getStartPosition(), node.getThenStatement().getLength());
					
					add_idx_tuple("If:test->body", node.getExpression().getStartPosition(), node.getExpression().getLength(), 
							node.getThenStatement().getStartPosition(), node.getThenStatement().getLength());
					
					if (node.getElseStatement() != null) {
						add_idx_tuple("If:test->orelse", node.getExpression().getStartPosition(), node.getExpression().getLength(), 
								node.getElseStatement().getStartPosition(), node.getElseStatement().getLength());
						
						add_idx_tuple("If:body->orelse", node.getThenStatement().getStartPosition(), node.getThenStatement().getLength(),
								 node.getElseStatement().getStartPosition(), node.getElseStatement().getLength());
					}
					return true;
				}
				
				
				public boolean visit(MethodInvocation node) {
					if (node.arguments().size()>0) {
						int length = ((ASTNode) node.arguments().get(node.arguments().size()-1)).getStartPosition() + 
								((ASTNode) node.arguments().get(node.arguments().size()-1)).getLength() - 
								((ASTNode) node.arguments().get(0)).getStartPosition();
						add_idx_tuple("Call:func->args", node.getName().getStartPosition(), node.getName().getLength(), 
								((ASTNode) node.arguments().get(0)).getStartPosition(), 
								length);
					}
					if (node.getExpression() != null) {
						add_idx_tuple("Attribute:value->attr", node.getExpression().getStartPosition(), node.getExpression().getLength(), 
								node.getName().getStartPosition(), node.getName().getLength());
					}
					return true;
				}
				
				public boolean visit(Assignment node) {
					add_idx_tuple("Assign:target->value", node.getLeftHandSide().getStartPosition(), node.getLeftHandSide().getLength(), node.getRightHandSide().getStartPosition(), node.getRightHandSide().getLength());
					return true;
				}
				
				
				public boolean visit(WhileStatement node) {
					if (!m_code.substring(node.getStartPosition(),node.getStartPosition()+5).equals("while")) {
						System.out.println("Error: 'while' is expected");
					}
					add_idx_tuple("While:while->test", node.getStartPosition(), 5 , 
							node.getExpression().getStartPosition(), node.getExpression().getLength());
					add_idx_tuple("While:while->body", node.getStartPosition(), 5, 
							node.getBody().getStartPosition(), node.getBody().getLength());
					add_idx_tuple("While:test->body", node.getExpression().getStartPosition(), node.getExpression().getLength(),
							node.getBody().getStartPosition(), node.getBody().getLength());
					
					return true;
				}
				
				public boolean visit(ForStatement node) {
					if (!m_code.substring(node.getStartPosition(),node.getStartPosition()+3).equals("for")) {
						System.out.println("Error: 'for' is expected");
					}
	                
					add_idx_tuple("For:for->body", node.getStartPosition(), 3, 
							node.getBody().getStartPosition(), node.getBody().getLength());
					if (node.getExpression() != null) {
						add_idx_tuple("For:for->test", node.getStartPosition(), 3, 
								node.getExpression().getStartPosition(), node.getExpression().getLength());
					}
					int initializerLength = 0;
					if (node.initializers().size()>0) {
						initializerLength = ((ASTNode) node.initializers().get(node.initializers().size()-1)).getStartPosition() + 
								((ASTNode) node.initializers().get(node.initializers().size()-1)).getLength() - 
								((ASTNode) node.initializers().get(0)).getStartPosition();
						add_idx_tuple("For:for->initializers", node.getStartPosition(), 3, 
								((ASTNode) node.initializers().get(0)).getStartPosition(), initializerLength);
					}
					int updaterLength = 0;
					if (node.updaters().size()>0) {
						updaterLength = ((ASTNode) node.updaters().get(node.updaters().size()-1)).getStartPosition() + 
								((ASTNode) node.updaters().get(node.updaters().size()-1)).getLength() - 
								((ASTNode) node.updaters().get(0)).getStartPosition();
						add_idx_tuple("For:for->updaters", node.getStartPosition(), 3, 
								((ASTNode) node.updaters().get(0)).getStartPosition(), updaterLength);
					}
					
					
					if (node.getExpression() != null) {
						if (initializerLength > 0)
							add_idx_tuple("For:initializers->test", ((ASTNode) node.initializers().get(0)).getStartPosition(), initializerLength,
									node.getExpression().getStartPosition(), node.getExpression().getLength());
						if (updaterLength > 0)
							add_idx_tuple("For:test->updaters", node.getExpression().getStartPosition(), node.getExpression().getLength(),
									((ASTNode) node.updaters().get(0)).getStartPosition(), updaterLength);
						add_idx_tuple("For:test->body", node.getExpression().getStartPosition(), node.getExpression().getLength(),
								node.getBody().getStartPosition(), node.getBody().getLength());
					}
					if (updaterLength > 0 && initializerLength > 0)
						add_idx_tuple("For:initializers->updaters", ((ASTNode) node.initializers().get(0)).getStartPosition(), initializerLength,
								((ASTNode) node.updaters().get(0)).getStartPosition(), updaterLength);
					if (initializerLength > 0)
						add_idx_tuple("For:initializers->body", ((ASTNode) node.initializers().get(0)).getStartPosition(), initializerLength,
								node.getBody().getStartPosition(), node.getBody().getLength());
					if (updaterLength > 0)
						add_idx_tuple("For:updaters->body", ((ASTNode) node.updaters().get(0)).getStartPosition(), updaterLength,
								node.getBody().getStartPosition(), node.getBody().getLength());
					
					return true;
				}
				
				public boolean visit(TryStatement node) {
					//				System.out.println(node);
					int length = 0;
					if (node.catchClauses().size() > 0)
						length = ((ASTNode) node.catchClauses().get(node.catchClauses().size()-1)).getStartPosition() + 
							((ASTNode) node.catchClauses().get(node.catchClauses().size()-1)).getLength() - 
							((ASTNode) node.catchClauses().get(0)).getStartPosition();
					if (node.getFinally() != null) {
						add_idx_tuple("Try:body->finalbody", node.getBody().getStartPosition(), node.getBody().getLength(), 
								node.getFinally().getStartPosition(), node.getFinally().getLength());
						if (length>0) 
							add_idx_tuple("Try:handler->finalbody", ((ASTNode) node.catchClauses().get(0)).getStartPosition(), length, 
									node.getFinally().getStartPosition(), node.getFinally().getLength());
					}
					if (length>0) {
						add_idx_tuple("Try:body->handler", node.getBody().getStartPosition(), node.getBody().getLength(), 
								((ASTNode) node.catchClauses().get(0)).getStartPosition(), length);
					}
					return true;
				}
				
				public boolean visit(ConditionalExpression node) {

					add_idx_tuple("IfExp:test->body", node.getExpression().getStartPosition(), node.getExpression().getLength(), 
								node.getThenExpression().getStartPosition(), node.getThenExpression().getLength());
					add_idx_tuple("IfExp:test->orelse", node.getExpression().getStartPosition(), node.getExpression().getLength(), 
							node.getElseExpression().getStartPosition(), node.getElseExpression().getLength());
					add_idx_tuple("IfExp:body->orelse", node.getThenExpression().getStartPosition(), node.getThenExpression().getLength(), 
							node.getElseExpression().getStartPosition(), node.getElseExpression().getLength());
					
					return true;
				}
				
				public boolean visit(DoStatement node) {
					if (!m_code.substring(node.getStartPosition(),node.getStartPosition()+2).equals("do")) {
						System.out.println("Error: 'do' is expected");
					}
					add_idx_tuple("Do:body->test", node.getBody().getStartPosition(), node.getBody().getLength(), 
								node.getExpression().getStartPosition(), node.getExpression().getLength());
					add_idx_tuple("Do:do->body", node.getStartPosition(), 2, 
							node.getBody().getStartPosition(), node.getBody().getLength());
					add_idx_tuple("Do:do->test", node.getStartPosition(), 2, 
							node.getExpression().getStartPosition(), node.getExpression().getLength());
					return true;
				}
				
				public boolean visit(SwitchStatement node) {
					if (!m_code.substring(node.getStartPosition(),node.getStartPosition()+6).equals("switch")) {
						System.out.println("Error: 'switch' is expected");
					}
					int length = ((ASTNode) node.statements().get(node.statements().size()-1)).getStartPosition() + 
							((ASTNode) node.statements().get(node.statements().size()-1)).getLength() - 
							((ASTNode) node.statements().get(0)).getStartPosition();
					add_idx_tuple("Switch:expr->statement", node.getExpression().getStartPosition(), node.getExpression().getLength(), 
							((ASTNode) node.statements().get(0)).getStartPosition(), length);
					add_idx_tuple("Switch:switch->expr", node.getStartPosition(), 6, 
							node.getExpression().getStartPosition(), node.getExpression().getLength());
					add_idx_tuple("Switch:switch->statement", node.getStartPosition(), 2, 
							((ASTNode) node.statements().get(0)).getStartPosition(), length);
					return true;
				}
				
				public boolean visit(ArrayAccess node) {
					add_idx_tuple("Subscript:value->slice", node.getArray().getStartPosition(), node.getArray().getLength(), 
							node.getIndex().getStartPosition(), node.getIndex().getLength());
					return true;
				}
				
				public boolean visit(LabeledStatement node) {
					add_idx_tuple("LabeledStatement:label->body", node.getLabel().getStartPosition(), node.getLabel().getLength(), 
							node.getBody().getStartPosition(), node.getBody().getLength());
					return true;
				}
				
				public boolean visit(InstanceofExpression node) {
					add_idx_tuple("InstanceofExpr:expr->type", node.getLeftOperand().getStartPosition(), node.getLeftOperand().getLength(), 
							node.getRightOperand().getStartPosition(), node.getRightOperand().getLength());
					return true;
				}
				
				public boolean visit(InfixExpression node) {
					add_idx_tuple("InfixExpr:left->right", node.getLeftOperand().getStartPosition(), node.getLeftOperand().getLength(), 
							node.getRightOperand().getStartPosition(), node.getRightOperand().getLength());
					return true;
				}
				
				public void preVisit(final ASTNode node) {
					if (node.getParent() != null) {
					add_idx_tuple("children:parent->child", node.getParent().getStartPosition(), node.getParent().getLength(), 
							node.getStartPosition(), node.getLength());
					}
				}
	
			});
			
			// append new labels to labels
			labels += "\n\n"+i+"\n"+String.join("\n", results);
			
			if (i%2000 == 0) {
				try {
					File myFile = new File("java_node_start_end_position.txt");
					FileWriter fileWriter = new FileWriter(myFile, append); // true to append
					                                                     // false to overwrite.
					fileWriter.write(labels);
					fileWriter.close();
				} catch (Exception e) {
					
				} 
				labels = "";
				append = true;
			}
		}
		try {
			File myFile = new File("java_node_start_end_position.txt");
			FileWriter fileWriter = new FileWriter(myFile, append); // true to append
			                                                     // false to overwrite.
			fileWriter.write(labels);
			fileWriter.close();
		} catch (Exception e) {
			
		} 
	}
}